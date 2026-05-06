"""
Step 2: Quality control using Kimi K2.5 (MoonShot API).
Checks all 400×6 candidates; flags failures for whole-group regeneration.
Supports checkpoint/resume.
"""
import json, time, argparse, os, re
import requests

MOONSHOT_URL = "https://api.moonshot.cn/v1/chat/completions"
MODEL = "kimi-k2.5"

SYSTEM_PROMPT = "你是一名财经翻译质检员。只返回JSON，不返回其他内容。"

QC_TEMPLATE = """你是一名财经翻译质检员。请检查以下候选译文是否满足实验要求。

实验要求：
这些候选译文应整体质量接近，不能存在明显误译、漏译、数字错误、主体错误、时间错误、涨跌方向错误、因果关系错误或明显不自然表达。候选之间允许存在语言特征差异，但不能存在明显质量梯度。

源文：
{source}

候选译文：
{candidates_json}

请逐条检查：
1. 是否保留源文核心信息。
2. 是否存在数字、时间、主体、方向、术语错误。
3. 是否存在明显漏译或增译。
4. 是否明显低于其他候选质量。
5. 是否符合其目标语言特征。

输出 JSON：
{{
  "overall_pass": true/false,
  "candidate_checks": [
    {{
      "id": "A",
      "pass": true/false,
      "problems": [],
      "quality_gap": "none/minor/major",
      "feature_match": "high/medium/low"
    }}
  ],
  "need_regeneration": ["..."],
  "reason": "..."
}}"""


def parse_json_response(text):
    text = text.strip()
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def call_qc(api_key, source, candidates, max_retries=5):
    """Call Kimi K2.5 for quality control check."""
    # Build candidates JSON for the prompt
    cand_json = json.dumps(
        [{'id': v['id'], 'feature_target': v['feature_target'], 'translation': v['translation']}
         for v in candidates],
        ensure_ascii=False, indent=2
    )
    user_prompt = QC_TEMPLATE.format(source=source, candidates_json=cand_json)

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    payload = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt}
        ],
        'max_tokens': 2048,
        'temperature': 0.3
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(MOONSHOT_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                body = resp.json()
                content = body['choices'][0]['message']['content']
                data = parse_json_response(content)
                return data, body.get('usage', {})
            elif resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f'  Rate limited, waiting {wait}s...')
                time.sleep(wait)
            elif resp.status_code >= 500:
                wait = min(2 ** attempt * 3, 30)
                print(f'  Server error {resp.status_code}, waiting {wait}s...')
                time.sleep(wait)
            else:
                last_error = f'HTTP {resp.status_code}: {resp.text[:200]}'
                print(f'  {last_error}')
                if resp.status_code == 400:
                    break
                time.sleep(2)
        except requests.exceptions.Timeout:
            print(f'  Timeout (attempt {attempt+1}/{max_retries})')
            time.sleep(3)
        except json.JSONDecodeError as e:
            print(f'  JSON parse error: {e}')
            time.sleep(2)
        except Exception as e:
            print(f'  Error: {e}')
            time.sleep(2)

    raise RuntimeError(f'QC failed after {max_retries} attempts: {last_error}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', required=True)
    parser.add_argument('--input', default='data/ec_400_with_candidates.json')
    parser.add_argument('--output', default='data/ec_400_qc_results.json')
    parser.add_argument('--delay', type=float, default=1.0)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    with open(args.input, encoding='utf-8') as f:
        candidates_data = json.load(f)
    print(f'Loaded {len(candidates_data)} samples from {args.input}')

    # Load checkpoint
    qc_results = []
    completed_ids = set()
    if os.path.exists(args.output):
        with open(args.output, encoding='utf-8') as f:
            qc_results = json.load(f)
        completed_ids = {r['sample_id'] for r in qc_results}
        print(f'Resuming: {len(qc_results)} already checked')

    limit = args.limit if args.limit > 0 else len(candidates_data)
    total = 0
    passed = 0
    failed = 0

    for entry in candidates_data[:limit]:
        sid = entry['sample_id']
        if sid in completed_ids:
            continue

        total += 1
        print(f'\n[{total}] Sample {sid} ...', end=' ', flush=True)

        try:
            qc_data, usage = call_qc(args.api_key, entry['source_text'], entry['variants'])

            qc_record = {
                'sample_id': sid,
                'overall_pass': qc_data.get('overall_pass', False),
                'candidate_checks': qc_data.get('candidate_checks', []),
                'need_regeneration': qc_data.get('need_regeneration', []),
                'reason': qc_data.get('reason', ''),
                'tokens': usage.get('total_tokens', 0)
            }
            qc_results.append(qc_record)
            completed_ids.add(sid)

            if qc_record['overall_pass']:
                print('PASS')
                passed += 1
            else:
                print(f'FAIL: {qc_record["reason"][:100]}')
                failed += 1

            # Save checkpoint
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(qc_results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f'ERROR: {e}')
            failed += 1

        if args.delay > 0:
            time.sleep(args.delay)

    pass_rate = passed / (passed + failed) * 100 if (passed + failed) > 0 else 0
    print(f'\n=== QC Summary ===')
    print(f'Checked: {total}')
    print(f'Passed: {passed}')
    print(f'Failed: {failed}')
    print(f'Pass rate: {pass_rate:.1f}%')
    print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
