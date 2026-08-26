$py = ".\.venv\Scripts\python.exe"
$runs = @(
  @{d="GQA-spatial"; s="caption"; t="GQA_caption"},
  @{d="GQA-spatial"; s="bridge";  t="GQA_bridge"},
  @{d="VQAv2";       s="caption"; t="VQAv2_caption"},
  @{d="VQAv2";       s="bridge";  t="VQAv2_bridge"},
  @{d="OK-VQA";      s="caption"; t="OKVQA_caption"},
  @{d="OK-VQA";      s="bridge";  t="OKVQA_bridge"}
)
foreach ($r in $runs) {
  Write-Host ("=== " + $r.t + " ===")
  & $py experiments\run_vqa.py --dataset $r.d --system $r.s --train-size 3000 --eval-size 400 --steps 1500 --batch-size 8 --lr 1e-3 --tag $r.t 2>&1 |
    Select-String -Pattern "host KV|step |\"accuracy|connector_params|train_seconds|peak_mem|loss_|saved" | ForEach-Object { $_.Line }
}
Write-Host "ALL MAIN RUNS DONE"
