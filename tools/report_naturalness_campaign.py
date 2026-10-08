"""Summarize measured diagnostics without declaring a subjective winner."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.vc.voice_library import digest


def main():
    out=ROOT/'recordings/v011_naturalness'
    result=json.loads((out/'metadata/results.json').read_text('utf-8'))
    rows=result['rows']
    ids=json.loads((out/'metadata/blind_manifest.json').read_text('utf-8'))['candidates']
    seed_path=out/'metadata/seed_attempt.json'
    seed=json.loads(seed_path.read_text('utf-8')) if seed_path.exists() else {'status':'PENDING'}
    stages=json.loads((out/'metadata/campaign.json').read_text('utf-8'))['stages']
    if seed_path.exists():stages=[s for s in stages if s['stage']!='seed']+[seed]
    before={'reference':'ba29c16a30675a92ca2ee9e19da9440315a4bd5002ab00f8a1ac6fb4bb59ab9e',
            'embedding':'0e53ecf0e0bd02592ae337979df017134559bab404c82314b49f136bed3dd508'}
    unchanged=(digest(ROOT/'models/meanvc2_ref60/reference.wav')==before['reference'] and
               digest(ROOT/'models/meanvc2_ref60/fixed_embedding.npy')==before['embedding'])
    settings=json.loads((ROOT/'settings.json').read_text('utf-8'))
    lines=['# 人間らしさ・幼さの改善比較','',
        'Status: READY FOR HUMAN LISTENING。自然さ・幼さ・好きな声の維持は未評価。自動Winnerは決めていない。','',
        f"匿名候補{len(ids)}本、波形が異なる候補{len(set(r['sha256'] for r in rows))}本。4つの既存Sourceを全条件へ使用。",
        'Referenceは現行＋中間域・低め域・高め域の5秒窓＋平均Embedding。抽出は音響指標のみで、落ち着き・力み・成人らしさは人間による確認が必要。',
        '補完なし／強弱のみ／強弱＋微小ピッチを同一の生成波形から比較。ReferenceとVC設定を同時に動かして結果を混同しない。','',
        '## アプリへの実装','',
        '- メイン・第2音声の「強弱のみ補完」を追加。既存Reference・モデル・待ち時間を維持し、ピッチ再編集をOFFにする。',
        '- 比較用Reference4種×3補完方式を声一覧に追加。現在の選択は自動で替えない。',
        '- 比較用「補完なし」は補完待ち1600msを省く。モデルbuffer1600ms＋補間40ms＋pre-roll640msは計2280ms。「補完あり」は計3880ms。I/Oと実機処理の変動を含む実測遅延ではない。',
        '- 先に実装済みの子音保護・音切れ復帰・無声域をまたがない補完を使用。さらに語頭・語尾の境界微分と急なF0変化を診断値として保存。根拠のない追加平滑化・一律Pitch/Formant/EQは掛けていない。','',
        '## 検証','',
        f"既存Reference・EmbeddingのSHA256維持: {unchanged}。現在の設定: {settings['ai_model']}。",
        '関連テスト: GUI/登録/補完48件、callback/ゲート/AI112件、計160件成功（重複したテストあり）。',
        '新しいnatural_lower_energyとnatural_lower_noneを実アプリと同じRPCで検証。固定Embedding一致、非有限値なし、既存モデルファイル一致。energyはpitch編集0。noneはPhraseRepairを作らない。',
        '全候補のclippingは0。Peak、声の高さの分位点、境界微分、5半音超の急変を保存。これらを自然さの自動順位には使わない。',
        '通常声の現行出力のvoiced F0中央値は約195Hz、Reference候補の中央値は227〜255Hz。単純に出力が高すぎると決めつけて全体ピッチを下げる根拠はない。低めReferenceが低め出力になるとも限らない。',
        'Offlineの補完は期限なしで全窓を処理する。実アプリは既存の非同期cacheを使い、間に合わない補正は元VC波形へbypassする。二つの聞こえ方の完全一致は保証していない。',
        '実マイク／Discord試験・人間聴取・実機End-to-End遅延測定は今回行っていない。','',
        '## 別モデルの品質上限比較','',
        f"Seed-VC V2: {seed['status']}。stageログ・RAM・最後のエラーはmetadata/seed_attempt.jsonに保存。",
        '現行Referenceの冒頭5秒を指定してもMeanVC2の声を完全に維持する保証はない。25秒までのReferenceによる試行は空きRAM1.5GiB下限で停止した。停止条件は緩めずReference長を短くして再試行した。N150での速度を確認しないままRealtimeアプリへ組み込まない。',
        '公式 inference_v2.py と既存隔離環境を使用。CPUはFP32、25steps、style変換OFF。モデル取得・cacheはDドライブ。','',
        '## 実行段階','', '| Stage | 結果 | 秒 | Peak RAM GiB | 理由 |','|---|---|---:|---:|---|']
    for s in stages:lines.append(f"| {s['stage']} | {s['status']} | {s['wall_seconds']:.2f} | {s['peak_ram_bytes']/1024**3:.2f} | {s.get('reason') or ''} |")
    if seed.get('last_error'):lines+=['','Seed最後のエラー:','```',seed['last_error'],'```']
    history=list((out/'metadata/attempt_history').glob('seed_*.json'))
    if history:
        lines+=['','Seed過去の試行（削除しない）:']
        for p in history:
            h=json.loads(p.read_text('utf-8'))
            lines.append(f"- {h['status']} / {h.get('reason')} / RAM {h['peak_ram_bytes']/1024**3:.2f}GiB / {h['log']}")
    if seed.get('status')=='SUCCESS' and (out/'metadata/seed_v2_result.json').exists():
        s=json.loads((out/'metadata/seed_v2_result.json').read_text('utf-8'))
        lines+=['',f"Seed-VC V2: generation {s['generation_seconds']:.2f}s / source {s['source_seconds']:.2f}s / RTF {s['rtf']:.2f} / {s['realtime_suitability']}。",
                f"model load {s['load_seconds']:.2f}s。声質・現在の声の維持はUNRATED。"]
    lines+=['','## 候補一覧（Blind対応表。試聴前は見なくてよい）','',
            '| Blind | Reference | Source | 補完 | Offline RTF（flush込） | Peak |','|---|---|---|---|---:|---:|']
    for r in ids:lines.append(f"| {r['id']} | {r['reference']} | {r['source']} | {r['mode']} | {r['rtf_including_flush']:.3f} | {r['peak']:.3f} |")
    lines+=['','比較ページ: recordings/v011_naturalness/blind/index.html。通常は長文15条件から表示。',
            '音量合わせ: DC除去と一定ゲインのみ。現在の声も同じ規則で合わせる。評価JSON: naturalness-ratings.json。',
            '再開: run_naturalness_campaign.ps1。成功済み生成は音声hashでcache判定。重い処理はCPU1コア、RAM上限4.5GiB、空きRAM1.5GiB・ディスク3GiBの監視付き。学習なし。',
            '元の4録音にない小声・相づち・笑い混じりの実発声は合成して捏造していない。追加録音での確認は今後の検証範囲。']
    destination=ROOT/'validation/v011/naturalness_campaign_report.md'
    destination.write_text('\n'.join(lines)+'\n','utf-8')
    print(str(destination))


if __name__=='__main__':main()
