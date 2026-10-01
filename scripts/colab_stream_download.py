"""Stream large artifacts through the authenticated Colab file endpoint.

Run using the Colab CLI's Python environment; never prints authentication data.
The CLI's normal download loads the entire base64 file into memory.
"""
import argparse
import os
from pathlib import Path
from urllib.parse import quote

import requests
from colab_cli.common import state
from colab_cli.contents import ContentsClient


def download(session, remote, local):
    client=ContentsClient(state.get_session(state.resolve_session(session)))
    target=Path(local)
    temporary=target.with_name(target.name+".partial")
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        with requests.get(client.base_url+"/files/"+quote(remote.strip("/"),safe="/"),
                          params={"authuser":"0","colab-runtime-proxy-token":client.token},
                          stream=True,timeout=(60,120)) as response:
            if response.status_code!=200:
                raise RuntimeError(f"Colab streaming download HTTP status {response.status_code}")
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    stream.write(chunk)
        os.replace(temporary,target)
        print(f"Recovered {target.stat().st_size} bytes to {target}",flush=True)
    except requests.RequestException:
        # Request exceptions include credential-bearing URLs; suppress them.
        raise RuntimeError("Colab streaming transfer failed; partial file retained") from None


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--session",required=True)
    parser.add_argument("remote")
    parser.add_argument("local")
    args=parser.parse_args()
    download(args.session,args.remote,args.local)
