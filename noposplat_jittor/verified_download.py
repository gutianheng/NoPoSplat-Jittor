"""Resumable bounded HTTPS byte ranges, full pinned SHA before publication."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request,urlopen,build_opener,ProxyHandler
from urllib.error import HTTPError,URLError
from http.client import IncompleteRead,RemoteDisconnected
from email.utils import parsedate_to_datetime
import hashlib,json,math,os,time,uuid,threading
from .dataset_conversion import atomic_json,sha256_file

_network_lock=threading.Lock()
_network_next=0.
_network_pause_until=0.


def is_transient_download_error(error):
    if isinstance(error,HTTPError):return error.code in (429,500,502,503,504)
    return isinstance(error,(URLError,TimeoutError,ConnectionError,IncompleteRead,RemoteDisconnected))


def retry_after_seconds(headers,attempt):
    value=headers.get('Retry-After') if headers else None
    if value:
        try:delay=float(value)
        except (ValueError,TypeError):
            try:delay=parsedate_to_datetime(value).timestamp()-time.time()
            except (ValueError,TypeError,OverflowError):delay=0.
        if math.isfinite(delay) and delay>0:return delay
    return min(90.,15.*2**attempt)


def _set_network_pause(delay):
    global _network_pause_until
    with _network_lock:_network_pause_until=max(_network_pause_until,time.monotonic()+delay)


def _wait_network_slot():
    global _network_next
    while True:
        with _network_lock:
            now=time.monotonic();wait=max(_network_pause_until,_network_next)-now
            if wait<=0:_network_next=now+1.2;return
        time.sleep(min(wait,5.))


def open_public_cdn_range(request,timeout,*,resolver=None,direct=None):
    """Resolve through the platform network route, then fetch HTTPS CDN directly.

    A signed range URL is used only for exactly the original requested range.
    It is never stored in the cache binding or printed. Every assembled source
    must still pass its original pinned SHA before publication.
    """
    origin=urlsplit(request.full_url)
    if origin.scheme!='https' or origin.hostname!='huggingface.co':
        raise ValueError('Split route requires the canonical public HTTPS source')
    resolve=resolver or urlopen
    with resolve(request,timeout=timeout) as response:
        if response.status!=206:raise ValueError('Official resolver did not honor ranged GET')
        final_url=response.geturl()
    final=urlsplit(final_url);host=final.hostname or ''
    trusted=host.endswith('.hf.co') or host.endswith('.huggingface.co') or host.endswith('.xethub.hf.co')
    if final.scheme!='https' or not trusted:raise ValueError('Unexpected public CDN redirect host')
    fetch=direct or build_opener(ProxyHandler({})).open
    return fetch(Request(final_url,headers=dict(request.header_items())),timeout=timeout)


def parallel_download(row,path,endpoint=None,workers=12,range_bytes=4*1024**2,*,open_url=None,split_route=False):
    if not 1<=workers<=16 or range_bytes<1:raise ValueError('Invalid bounded range concurrency/size')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.stat().st_size!=row['bytes'] or sha256_file(path)!=row['sha256']:raise ValueError('Existing raw chunk differs from source hash')
        return
    if path.with_name(path.name+'.part').exists():raise ValueError('Sequential partial exists; keep it or use a separate range cache')
    if split_route and (endpoint is not None or open_url is not None):raise ValueError('Split route cannot combine a mirror or injected opener')
    url=row['url']
    if endpoint is not None:
        if endpoint!='https://hf-mirror.com' or urlsplit(url).netloc!='huggingface.co' or urlsplit(url).scheme!='https':raise ValueError('Unexpected mirror/source host')
        url=url.replace('https://huggingface.co/',endpoint+'/',1)
    parts=path.with_name(path.name+'.ranges');marker=parts/'source.json'
    binding={k:row[k] for k in ('url','bytes','sha256')};binding['range_bytes']=range_bytes
    if parts.exists() and not marker.exists():raise FileExistsError('Unowned range directory')
    parts.mkdir(exist_ok=True)
    if marker.exists():
        if json.loads(marker.read_text())!=binding:raise ValueError('Range cache belongs to another pinned source')
    else:atomic_json(marker,binding)
    opener=open_public_cdn_range if split_route else (open_url or urlopen)
    def fetch(i):
        if path.exists():return  # A separate bounded transport may publish the identical full source.
        lo=i*range_bytes;hi=min(row['bytes'],lo+range_bytes)-1;final=parts/f'{i:06d}.bin'
        if final.exists():
            if final.stat().st_size!=hi-lo+1:raise ValueError('Wrong completed range size')
            return
        temp=final.with_suffix('.partial')
        for attempt in range(8):
            try:
                if path.exists():return
                if open_url is None:_wait_network_slot()
                request=Request(url,headers={'Range':f'bytes={lo}-{hi}','User-Agent':'NoPoSplat-verified-ranges/1.0'})
                with opener(request,timeout=60) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{row["bytes"]}':raise ValueError('Server did not honor exact requested range')
                    if hasattr(response,'read1'):
                        blocks=[];received=0;started=time.monotonic()
                        while received<hi-lo+2:
                            if path.exists():return
                            block=response.read1(min(256*1024,hi-lo+2-received))
                            if not block:break
                            blocks.append(block);received+=len(block)
                            if time.monotonic()-started>180:raise TimeoutError('Range body exceeded bounded wall time')
                        if path.exists():return
                        payload=b''.join(blocks)
                    else:payload=response.read(hi-lo+2)
                    if len(payload)!=hi-lo+1:raise ValueError('Wrong HTTP range payload length')
                with temp.open('wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
                os.replace(temp,final);return
            except HTTPError as error:
                if error.code in (429,500,502,503,504):
                    if attempt==7:raise
                    delay=retry_after_seconds(error.headers,attempt);_set_network_pause(delay)
                    print(json.dumps({'network_status':'server_rate_limit' if error.code==429 else 'temporary_server_error',
                        'http_status':error.code,'retry_after_seconds':delay,'attempt':attempt+1}),flush=True)
                    continue
                if attempt>=2:raise
                time.sleep(2**attempt)
            except (OSError,TimeoutError,IncompleteRead,RemoteDisconnected):
                if attempt>=2:raise
                time.sleep(2**attempt)
    count=math.ceil(row['bytes']/range_bytes)
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(fetch,range(count)))
    if path.exists():
        if path.stat().st_size!=row['bytes'] or sha256_file(path)!=row['sha256']:raise ValueError('Published assisted source differs from pinned SHA')
        if json.loads(marker.read_text())!=binding:raise ValueError('Owned range binding changed')
        for i in range(count):
            for file in (parts/f'{i:06d}.bin',parts/f'{i:06d}.partial'):
                if file.is_symlink():raise ValueError('Unexpected symlink in owned range cache')
                if file.exists():file.unlink()
        marker.unlink();parts.rmdir();return
    assembled=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.assembling');digest=hashlib.sha256()
    try:
        with assembled.open('wb') as f:
            for i in range(count):
                data=(parts/f'{i:06d}.bin').read_bytes();digest.update(data);f.write(data)
            f.flush();os.fsync(f.fileno())
        if assembled.stat().st_size!=row['bytes'] or digest.hexdigest()!=row['sha256']:raise ValueError('Assembled source SHA256 mismatch; refusing publication/deserialization')
        os.replace(assembled,path)
        for i in range(count):(parts/f'{i:06d}.bin').unlink()
        marker.unlink();parts.rmdir()
    finally:
        if assembled.exists():assembled.unlink()
