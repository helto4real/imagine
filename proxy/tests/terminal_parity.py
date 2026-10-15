#!/usr/bin/env python3
"""PTY/redirected Strata filter and bootstrap parity with synthetic bytes."""
import argparse,errno,json,os,pathlib,pty,subprocess,sys,tempfile,termios
p=argparse.ArgumentParser();p.add_argument('--reference',type=pathlib.Path,required=True);a=p.parse_args();ROOT=pathlib.Path(__file__).resolve().parents[2]
data=b'[strata] starting the engine\n[strata] done: 38 tokens in 29 s (51.9 tok/s) (stop, cancel=False), expert cache 39.1% hit\r\n[strata] \x1b[31merror\x1b[0m\nraw \xff\xfe\n'+chr(0xf0000).encode()+b'\n'
def run(kind,tty,nc,term):
 env={**os.environ,'PYTHONPATH':str(a.reference/'src'),'PYTHONDONTWRITEBYTECODE':'1','TERM':term};env.pop('NO_COLOR',None)
 if nc is not None:env['NO_COLOR']=nc
 cmd=[sys.executable,'-m','st_proxy.strata_logging'] if kind=='python' else [str(ROOT/'target/debug/st-strata-logging')]
 if not tty:return subprocess.run(cmd,input=data,stdout=subprocess.PIPE,env=env,check=True).stdout
 master,slave=pty.openpty();attrs=termios.tcgetattr(slave);attrs[1]&=~termios.OPOST;termios.tcsetattr(slave,termios.TCSANOW,attrs)
 process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=slave,env=env);os.close(slave);process.stdin.write(data);process.stdin.close();result=b''
 while True:
  try:chunk=os.read(master,65536)
  except OSError as e:
   if e.errno==errno.EIO:break
   raise
  if not chunk:break
  result+=chunk
 os.close(master);assert process.wait()==0;return result
count=0
for tty in (False,True):
 for nc in (None,'','1'):
  for term in ('xterm-256color','dumb'):
   python=run('python',tty,nc,term);rust=run('rust',tty,nc,term);assert python==rust,(tty,nc,term,python,rust);count+=1
print(f'PASS: {count} exact terminal filter comparisons, including non-UTF8 and private-use Unicode')
