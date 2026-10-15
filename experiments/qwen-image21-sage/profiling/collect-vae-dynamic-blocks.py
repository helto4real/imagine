"""Summarize only the synthetic benchmark output selected on the command line."""
from pathlib import Path
import argparse,hashlib,json,re
from PIL import Image,ImageChops
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=root/'evidence/vae-dynamic-blocks'
results=json.loads((a.run/'results.json').read_text());summary={};comparisons={}
for key,row in results.items():
 assert row['returncode']==0,(key,row)
 path=Path(row['path']);items=[]
 for r in row['runs']:
  text=(path/f"native-request-{r['index']}.log").read_text()
  tiles=re.findall(r'processing (\d+) tiles',text)
  sizes=re.findall(r'VAE decode tile size: (\d+x\d+) pixels',text)
  items.append({**r,'final_vae_tiles':int(tiles[-1]) if tiles else 1,'final_vae_tile_size':sizes[-1] if sizes else 'full image','dynamic_upscale_events':text.count('VAE dynamic upscale-conv:')})
 summary[key]={'run':path.name,'rgba':row['rgba'],'repeat_bitidentical':len(set(row['rgba']))==1,'results':items}
for w in [1600,4096]:
 paths=[Path(results[f'image-{w}-{mode}']['path'])/'native-image-1.png' for mode in ['control','candidate']]
 images=[Image.open(p).convert('RGBA') for p in paths];diff=ImageChops.difference(*images);channels=diff.split()
 mask=channels[0]
 for channel in channels[1:]:mask=ImageChops.lighter(mask,channel)
 pixels=images[0].width*images[0].height;different=pixels-mask.histogram()[0]
 rgb_hist=diff.convert('RGB').histogram();mae=sum((i%256)*n for i,n in enumerate(rgb_hist))/(pixels*3)
 comparisons[str(w)]={'bitidentical':different==0,'different_pixels':different,'total_pixels':pixels,'different_pixel_fraction':different/pixels,'rgb_mean_abs_error_0_255':mae,'max_channel_error':max(c.getextrema()[1] for c in channels)}
(out/'image-results.json').write_text(json.dumps({'coordinator':a.run.name,'series':summary,'warm_image_comparisons':comparisons},indent=2)+'\n')
print(json.dumps(comparisons,indent=2))
