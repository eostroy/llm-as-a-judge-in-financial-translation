"""
Step 3: Judge ranking — 4 models × 3 repetitions × 400 samples.
Each rep independently shuffles A-F → T1-T6 for blind ranking.
Outputs rankings + T-mappings for each rep.
Supports per-model checkpoint/resume.
"""
import json, time, argparse, os, re, random
import requests
from collections import defaultdict

# Judge model configurations
JUDGE_MODELS = {
    'gpt-5.5': {
        'provider': 'openrouter',
        'url': 'https://openrouter.ai/api/v1/chat/completions',
        'model': 'openai/gpt-5.5',
        'headers': lambda key: {
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'http://localhost',
            'X-Title': 'LLM-as-Judge-Ranking'
        }
    },
    'gemini-3.1-flash-lite': {
        'provider': 'openrouter',
        'url': 'https://openrouter.ai/api/v1/chat/completions',
        'model': 'google/gemini-3.1-flash-lite-preview',
        'headers': lambda key: {
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'http://localhost',
            'X-Title': 'LLM-as-Judge-Ranking'
        }
    },
    'deepseek-v4-flash': {
        'provider': 'deepseek',
        'url': 'https://api.deepseek.com/v1/chat/completions',
        'model': 'deepseek-v4-flash',
        'headers': lambda key: {
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json'
        }
    },
    'kimi-k2.6': {
        'provider': 'moonshot',
        'url': 'https://api.moonshot.cn/v1/chat/completions',
        'model': 'kimi-k2.6',
        'headers': lambda key: {
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json'
        }
    }
}

SYSTEM_PROMPT = "你是一名专业财经翻译评审。只返回JSON，不要其他内容。"

RANKING_TEMPLATE = """你是一名专业财经翻译评审。请根据源文，对 6 条候选译文进行质量排序。

候选译文可能在表达方式上存在差异，但这些差异不必然代表质量高低，也不代表应该偏向有差异的表达。请优先判断译文是否准确、完整、自然、符合财经语境。若多条译文均准确自然，再根据整体语言组织选择更适合作为正式财经译文的一条。

源文：
{source}

候选译文：
T1: {t1}
T2: {t2}
T3: {t3}
T4: {t4}
T5: {t5}
T6: {t6}

请返回JSON对象，ranking数组包含T1到T6的完整排序，从最佳到最差。

例如：{{"ranking": ["T3", "T1", "T5", "T2", "T6", "T4"]}}"""


def parse_json_response(text):
    text = text.strip()
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def shuffle_candidates(variants):
    """Shuffle A-F → T1-T6. Returns (t_map, t_variants)."""
    letters = [v['id'] for v in variants]
    shuffled = letters[:]
    random.shuffle(shuffled)
    t_map = {}
    reverse_map = {}
    for i, letter in enumerate(shuffled):
        t_label = f'T{i+1}'
        t_map[t_label] = letter
        reverse_map[letter] = t_label
    # Build T1-T6 list
    t_variants = []
    for i in range(6):
        t_label = f'T{i+1}'
        letter = t_map[t_label]
        trans = next(v['translation'] for v in variants if v['id'] == letter)
        t_variants.append((t_label, letter, trans))
    return t_map, reverse_map, t_variants


def call_judge(judge_key, api_key, source, t_variants, max_retries=5):
    """Call a judge model to rank T1-T6."""
    config = JUDGE_MODELS[judge_key]
    headers = config['headers'](api_key)

    # Build prompt
    args = {'source': source}
    for t_label, letter, trans in t_variants:
        args[t_label.lower()] = trans

    user_prompt = RANKING_TEMPLATE.format(**args)

    payload = {
        'model': config['model'],
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
            resp = requests.post(config['url'], headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                body = resp.json()
                content = body['choices'][0]['message']['content']
                data = parse_json_response(content)
                return data, body.get('usage', {})
            elif resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f'    Rate limited, waiting {wait}s...')
                time.sleep(wait)
            elif resp.status_code >= 500:
                wait = min(2 ** attempt * 3, 30)
                print(f'    Server error {resp.status_code}, waiting {wait}s...')
                time.sleep(wait)
            else:
                last_error = f'HTTP {resp.status_code}: {resp.text[:200]}'
                print(f'    {last_error}')
                if resp.status_code == 400:
                    break
                time.sleep(2)
        except requests.exceptions.Timeout:
            print(f'    Timeout (attempt {attempt+1}/{max_retries})')
            time.sleep(3)
        except json.JSONDecodeError as e:
            print(f'    JSON parse error: {e}')
            time.sleep(2)
        except Exception as e:
            print(f'    Error: {e}')
            time.sleep(2)

    raise RuntimeError(f'Ranking failed after {max_retries} attempts: {last_error}')


def load_or_init_output(output_path):
    if os.path.exists(output_path):
        with open(output_path, encoding='utf-8') as f:
            return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--openrouter-key', required=True)
    parser.add_argument('--deepseek-key', required=True)
    parser.add_argument('--moonshot-key', required=True)
    parser.add_argument('--input', default='data/ec_400_with_candidates.json')
    parser.add_argument('--output-dir', default='data/rankings')
    parser.add_argument('--judges', nargs='+', default=['gpt-5.5', 'gemini-3.1-flash-lite', 'deepseek-v4-flash', 'kimi-k2.6'])
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--delay', type=float, default=1.0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Build API key map
    keys = {
        'openrouter': args.openrouter_key,
        'deepseek': args.deepseek_key,
        'moonshot': args.moonshot_key
    }

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input, encoding='utf-8') as f:
        candidates_data = json.load(f)
    print(f'Loaded {len(candidates_data)} samples')

    limit = args.limit if args.limit > 0 else len(candidates_data)

    for judge_key in args.judges:
        config = JUDGE_MODELS[judge_key]
        api_key = keys[config['provider']]
        print(f'\n{"="*60}')
        print(f'Judge: {judge_key} ({config["model"]}) via {config["provider"]}')
        print(f'{"="*60}')

        for rep in range(1, args.reps + 1):
            # Use different seed for each rep
            random.seed(args.seed * 1000 + rep * 100 + hash(judge_key) % 1000)

            output_path = os.path.join(args.output_dir, f'rankings_{judge_key}_rep{rep}.json')
            results = load_or_init_output(output_path)
            completed_ids = {r['sample_id'] for r in results}
            print(f'\nRep {rep}/{args.reps}: {len(results)} already ranked')

            for entry in candidates_data[:limit]:
                sid = entry['sample_id']
                if sid in completed_ids:
                    continue

                # Shuffle candidates
                t_map, reverse_map, t_variants = shuffle_candidates(entry['variants'])

                print(f'  Sample {sid} ...', end=' ', flush=True)

                try:
                    ranking_data, usage = call_judge(judge_key, api_key, entry['source_text'], t_variants)

                    # Convert T-ranks to letter-ranks
                    t_ranking = ranking_data.get('ranking', [])
                    letter_ranking = [t_map.get(t, t) for t in t_ranking]

                    record = {
                        'sample_id': sid,
                        'rep': rep,
                        't_map': t_map,  # T1 → A
                        'reverse_map': reverse_map,  # A → T1
                        't_ranking': t_ranking,
                        'letter_ranking': letter_ranking,
                        'tokens': usage.get('total_tokens', 0)
                    }
                    results.append(record)
                    completed_ids.add(sid)

                    print(f'OK: {letter_ranking}')

                    # Save checkpoint
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)

                except Exception as e:
                    print(f'FAIL: {e}')

                if args.delay > 0:
                    time.sleep(args.delay)

            print(f'Rep {rep} done: {len(results)} total rankings')

    print('\n=== Ranking complete ===')


if __name__ == '__main__':
    main()
