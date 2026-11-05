"""Generate evidence ONLY from actual CSV/JSON/PNG outputs; no fabricated curves."""
from pathlib import Path
import argparse
import csv
import html
import json
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();root=Path(args.run)
    report=['# 真实运行记录汇总','本页只汇总此目录实际生成的记录；不自动宣称达到论文精度。']
    if (root/'train.csv').exists():
        with (root/'train.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        if rows:
            x=np.array([int(r['step']) for r in rows]);y=np.array([float(r['loss']) for r in rows])
            if not np.isfinite(y).all():raise ValueError('Refusing to plot nonfinite loss as a valid run')
            xp=60+(x-x.min())/max(int(x.max()-x.min()),1)*800
            yp=340-(y-y.min())/max(float(y.max()-y.min()),1e-12)*280
            points=' '.join(f'{a:.2f},{b:.2f}' for a,b in zip(xp,yp))
            svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="920" height="400" viewBox="0 0 920 400"><rect width="920" height="400" fill="white"/><text x="60" y="26" font-size="18">Actual training loss</text><path d="M60 60V340H860" fill="none" stroke="#555"/><polyline points="{points}" stroke="#176d84" stroke-width="2" fill="none"/><text x="60" y="380">Steps {x.min()} to {x.max()}; loss {y.min():.6g} to {y.max():.6g}</text></svg>'
            (root/'loss.svg').write_text(svg,encoding='utf-8');report+=['','![实际训练损失](loss.svg)',f'日志步数：{len(rows)}；最后一步：{x[-1]}。']
    if (root/'results.json').exists():
        data=json.loads((root/'results.json').read_text(encoding='utf-8'))
        report+=['',f'场景数：{data["scene_count"]}；目标图数：{data["target_count"]}；相机对齐：{data["align_pose"]}。','','|聚合方式|PSNR↑|SSIM↑|LPIPS↓|','|---|---:|---:|---:|']
        for key in ('scene_mean','image_mean'):
            r=data[key];report.append(f'|{key}|{r["psnr"]:.4f}|{r["ssim"]:.5f}|{r["lpips"]:.5f}|')
    renders=sorted((root/'images').rglob('*_render.png'))[:8]
    if renders:
        pairs=[]
        for render in renders:
            gt=render.with_name(render.name.replace('_render.png','_gt.png'))
            if gt.exists():pairs.append((Image.open(gt).convert('RGB'),Image.open(render).convert('RGB')))
        if pairs:
            w,h=pairs[0][0].size;sheet=Image.new('RGB',(w*2,h*len(pairs)))
            for i,(a,b) in enumerate(pairs):sheet.paste(a,(0,i*h));sheet.paste(b,(w,i*h))
            sheet.save(root/'comparison.png');report+=['','左GT，右Jittor渲染；按文件名选取前8对，不声称是随机代表样本。','','![渲染对照](comparison.png)']
    (root/'运行汇总.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(root/'运行汇总.md')


if __name__=='__main__':main()
