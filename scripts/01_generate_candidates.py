"""
Step 1: Generate 6 candidates (A-F) for 400 EC samples using Claude Sonnet 4.6 via OpenRouter.
One API call per sample — all 6 candidates generated together.
Supports checkpoint/resume.
"""
import json, time, argparse, os, sys, re, logging
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-sonnet-4.6"

# Setup logging for debugging parse failures
LOG_DIR = 'data/logs'
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, 'generation_errors.log'),
                    level=logging.WARNING,
                    format='%(asctime)s %(message)s')

SYSTEM_PROMPT = "你是一名专业财经翻译专家。只返回JSON，不返回其他内容。"

USER_PROMPT_TEMPLATE = """你是一名专业财经翻译专家。请将以下英文源文翻译为 6 条中文译文。6 条译文必须：
- 语义准确，完整保留数字、时间、主体、涨跌方向、因果关系和专业术语。
- 整体质量相同，均可作为可交付的财经译文。差异仅来自语言组织方式。
- 每条译文独立完整，不得在译文中添加解释性括号或说明。

源文（英文）：
{source}

请生成以下 6 个版本：

A_baseline_balanced：
准确、自然、平衡的译文，符合中文财经报道常见写法。不过度意译、不刻意书面化、不刻意贴近源文结构。

B_source_syntax_preserving：
尽量保留源文的句法主干、信息顺序和修饰关系，但仍需符合中文基本表达习惯。

C_target_language_restructured：
按照中文财经文本习惯调整信息顺序。可以将英语中后置的原因、条件、时间状语提前，使表达更符合中文行文方式。调整应限于局部语序优化，不应大幅改变源文整体段落结构和信息推进节奏。

D_cohesion_explicit：
适度显化句内或句间逻辑关系，如因果、转折、递进、指代、概括等。不得添加源文没有的事实信息。

E_formal_register：
提高书面化程度和财经报道语域特征，使用更正式、更规范的现代中文词汇和表达。不得改变事实含义。必须全部使用现代中文，严格禁止出现任何英文词汇。严格禁止使用文言或近文言的表达（包括但不限于"之/彼/然/业已/遂/逮/哉/乎/矣/焉/耳/且夫/盖/伏惟/窃以为/岂/曷/讵/胡/奚"等），也不得使用"函/牍/兹/兹因/据此/查/呈/奉/尚"等旧式公文套话。可以用的正式表达包括：较正式的双音节词、财经领域术语、规范的书面句式，但整体必须读起来像现代财经新闻报道。

F_information_unpacking：
有针对性地拆分原文中特别密集的结构（如多重定语从句嵌套、长修饰链、复杂名词短语），使信息更易理解。适度分句即可，不需要将所有信息拆成细碎短句。不得添加源文没有的事实、背景、解释或过渡词。整体节奏应接近现代财经报道的正常行文，不要写成新闻简讯或儿童读物。

硬性要求：
1. 6 条译文质量必须非常接近，不能让某一条明显优于或劣于其他。
2. 差异必须主要来自指定语言特征，而非翻译质量本身。
3. 输出必须是合法 JSON。

输出格式：
{{
  "variants": [
    {{"id": "A", "feature_target": "baseline_balanced", "translation": "..."}},
    {{"id": "B", "feature_target": "source_syntax_preserving", "translation": "..."}},
    {{"id": "C", "feature_target": "target_language_restructured", "translation": "..."}},
    {{"id": "D", "feature_target": "cohesion_explicit", "translation": "..."}},
    {{"id": "E", "feature_target": "formal_register", "translation": "..."}},
    {{"id": "F", "feature_target": "information_unpacking", "translation": "..."}}
  ]
}}"""


def parse_json_response(text):
    """Extract JSON from model response, handling markdown code blocks and resilient to
    unescaped ASCII double-quotes in Chinese translation values."""
    original = text.strip()

    # Remove markdown code block markers
    cleaned = re.sub(r'```\w*', '', original)
    cleaned = re.sub(r'```', '', cleaned)
    cleaned = cleaned.strip()

    # Try direct parsing first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Resilient extraction: find each variant block by locating {"id": "X" ...} patterns
    # Pattern matches: "id": "X", "feature_target": "...", "translation": "..." where translation
    # may contain unescaped ASCII double-quotes (from Chinese quotation marks rendered as ")
    variant_re = re.compile(
        r'"id"\s*:\s*"([A-F])"\s*,\s*'
        r'"feature_target"\s*:\s*"([^"]*)"\s*,\s*'
        r'"translation"\s*:\s*"',
        re.DOTALL
    )

    variants = []
    pos = 0
    while pos < len(cleaned):
        m = variant_re.search(cleaned, pos)
        if not m:
            break
        vid = m.group(1)
        ft = m.group(2)
        trans_start = m.end()
        # Find the closing quote of translation value
        # Walk forward, handling escape sequences, until we find an unescaped quote
        # followed by optional whitespace and } or ,
        trans_end = trans_start
        while trans_end < len(cleaned):
            c = cleaned[trans_end]
            if c == '\\':
                trans_end += 2  # skip escaped char
                continue
            elif c == '"':
                # Check if this quote is followed by end of object or array
                after = cleaned[trans_end+1:trans_end+20].lstrip()
                if after.startswith('}') or after.startswith('],') or after.startswith(']'):
                    break
                # Otherwise it's a quote inside the translation - skip it
                trans_end += 1
                continue
            trans_end += 1
        translation = cleaned[trans_start:trans_end]
        # Unescape JSON escapes
        translation = translation.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
        variants.append({
            'id': vid,
            'feature_target': ft,
            'translation': translation
        })
        pos = trans_end + 1

    if len(variants) == 6 and {v['id'] for v in variants} == {'A', 'B', 'C', 'D', 'E', 'F'}:
        return {'variants': variants}

    # Save failing response for debugging
    err_log = os.path.join(LOG_DIR, 'parse_failures.jsonl')
    with open(err_log, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'raw': original, 'variant_count': len(variants)}, ensure_ascii=False) + '\n')

    raise json.JSONDecodeError(f'Parse failed: got {len(variants)} variants', original, 0)


def validate_variants(data, sample_id):
    """Check that variants contain all 6 required IDs."""
    variants = data.get('variants', [])
    ids = {v['id'] for v in variants}
    expected = {'A', 'B', 'C', 'D', 'E', 'F'}
    missing = expected - ids
    extra = ids - expected
    errors = []
    if missing:
        errors.append(f'Missing variants: {missing}')
    if extra:
        errors.append(f'Extra variants: {extra}')
    for v in variants:
        if not v.get('translation', '').strip():
            errors.append(f'Empty translation for {v.get("id", "?")}')
    return len(errors) == 0, errors


def call_api(api_key, source_text, max_retries=5):
    """Call OpenRouter API to generate 6 candidates for one source."""
    user_prompt = USER_PROMPT_TEMPLATE.format(source=source_text)

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'http://localhost',
        'X-Title': 'LLM-as-Judge-EC-Experiment'
    }

    payload = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt}
        ],
        'max_tokens': 8192,
        'temperature': 0.3
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=180)
            if resp.status_code == 200:
                body = resp.json()
                content = body['choices'][0]['message']['content']
                data = parse_json_response(content)
                return data, body.get('model', ''), body.get('usage', {})
            elif resp.status_code == 429:
                wait = min(2 ** attempt * 5, 120)
                print(f'  Rate limited, waiting {wait}s...')
                time.sleep(wait)
            elif resp.status_code >= 500:
                wait = min(2 ** attempt * 3, 60)
                print(f'  Server error {resp.status_code}, waiting {wait}s...')
                time.sleep(wait)
            else:
                last_error = f'HTTP {resp.status_code}: {resp.text[:300]}'
                print(f'  {last_error}')
                if resp.status_code == 400:
                    break  # Don't retry bad requests
                time.sleep(2)
        except requests.exceptions.Timeout:
            print(f'  Timeout (attempt {attempt+1}/{max_retries})')
            time.sleep(5)
        except json.JSONDecodeError as e:
            print(f'  JSON parse error: {e}')
            time.sleep(2)
        except Exception as e:
            print(f'  Unexpected error: {e}')
            time.sleep(3)

    raise RuntimeError(f'Failed after {max_retries} attempts. Last error: {last_error}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', required=True, help='OpenRouter API key')
    parser.add_argument('--input', default='data/ec_400_sources.json')
    parser.add_argument('--output', default='data/ec_400_with_candidates.json')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between requests (seconds)')
    parser.add_argument('--start', type=int, default=0, help='Start from this index (0-based)')
    parser.add_argument('--limit', type=int, default=0, help='Max samples to process (0=all)')
    args = parser.parse_args()

    # Load sources
    with open(args.input, encoding='utf-8') as f:
        sources = json.load(f)
    print(f'Loaded {len(sources)} sources from {args.input}')

    # Load existing checkpoint
    results = []
    completed_ids = set()
    if os.path.exists(args.output):
        with open(args.output, encoding='utf-8') as f:
            results = json.load(f)
        completed_ids = {r['sample_id'] for r in results}
        print(f'Resuming: {len(results)} already completed')

    # Determine range
    end_idx = len(sources) if args.limit == 0 else min(args.start + args.limit, len(sources))

    total = 0
    success = 0
    fail = 0

    for i in range(args.start, end_idx):
        src = sources[i]
        sid = src['sample_id']

        if sid in completed_ids:
            continue

        total += 1
        print(f'\n[{i+1}/{len(sources)}] Sample {sid} ...', end=' ', flush=True)

        try:
            data, model_used, usage = call_api(args.api_key, src['source_text'])
            ok, errors = validate_variants(data, sid)

            if ok:
                entry = {
                    'sample_id': sid,
                    'index': i,
                    'source_text': src['source_text'],
                    'variants': data['variants'],
                    'model_used': model_used,
                    'usage': usage,
                    'generation_round': 1
                }
                results.append(entry)
                completed_ids.add(sid)
                success += 1

                tokens = usage.get('total_tokens', '?')
                print(f'OK (tokens={tokens})')

                # Save checkpoint
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            else:
                print(f'VALIDATION FAIL: {errors}')
                fail += 1

        except Exception as e:
            print(f'FAIL: {e}')
            fail += 1

        # Rate limit delay
        if args.delay > 0:
            time.sleep(args.delay)

    print(f'\n=== Summary ===')
    print(f'Processed: {total}')
    print(f'Success: {success}')
    print(f'Failed: {fail}')
    print(f'Total in output: {len(results)}')
    print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
