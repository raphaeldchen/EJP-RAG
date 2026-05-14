# Audit App Deployment

**Date deployed:** 2026-05-13/14
**Status:** Live at https://ejp-rag-audit.com

## What was deployed

`audit_app.py` — a Streamlit app for lawyers to label retrieval results (BINDING / RELEVANT / IRRELEVANT) to generate ground-truth data for embedding and reranker fine-tuning. It runs the full production retrieval stack (hybrid vector + BM25 + CrossEncoder reranker) so labels reflect what the production system actually surfaces.

## Infrastructure

| Resource | Details |
|---|---|
| Cloud provider | Oracle Cloud (OCI) — trial credit (~$300, lasts ~Oct/Nov 2026) |
| Shape | VM.Standard3.Flex (Intel) — 1 OCPU, 16 GB RAM |
| OS | Ubuntu 22.04 |
| Public IP | 163.192.97.229 |
| Domain | ejp-rag-audit.com (Squarespace, ~$14/yr, auto-renews) |
| DNS | Cloudflare (free, nameservers set in Squarespace) |
| HTTPS | Cloudflare Tunnel (cloudflared.service) |

## SSH access

```bash
ssh ubuntu@163.192.97.229
```

## Server file layout

```
/home/ubuntu/legal_rag/                    # git repo
/home/ubuntu/legal_rag/.env               # secrets (chmod 600, never commit)
/etc/cloudflared/config.yml               # tunnel ingress config
/etc/systemd/system/audit-app.service     # Streamlit service
/etc/systemd/system/cloudflared.service   # tunnel service (auto-installed)
/etc/systemd/system/ollama.service        # Ollama service (auto-installed)
```

## Environment variables (`/home/ubuntu/legal_rag/.env`)

```
SUPABASE_URL=https://srxvkrfmjwtunevkbkye.supabase.co
SUPABASE_SERVICE_KEY=<service key>
ADMIN_PASSWORD=<admin password>
ANTHROPIC_API_KEY=<anthropic key>
OLLAMA_BASE_URL=http://localhost:11434
```

## Services

Three systemd services run permanently:

```bash
sudo systemctl status audit-app      # Streamlit on port 8501
sudo systemctl status ollama         # Ollama (nomic-embed-text + llama3.2)
sudo systemctl status cloudflared    # HTTPS tunnel
```

### audit-app.service

```ini
[Unit]
Description=Legal RAG Audit App
After=network.target ollama.service
Wants=ollama.service

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/legal_rag
EnvironmentFile=/home/ubuntu/legal_rag/.env
ExecStart=/home/ubuntu/legal_rag/venv/bin/streamlit run audit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Cloudflare tunnel config (`/etc/cloudflared/config.yml`)

```yaml
tunnel: af2decc7-ae74-4d88-b916-0c7cca7a8a58
credentials-file: /etc/cloudflared/af2decc7-ae74-4d88-b916-0c7cca7a8a58.json

ingress:
  - hostname: ejp-rag-audit.com
    service: http://localhost:8501
  - hostname: www.ejp-rag-audit.com
    service: http://localhost:8501
  - service: http_status:404
```

## Cloudflare DNS records

| Type | Name | Target | Proxy |
|---|---|---|---|
| CNAME | `@` | `af2decc7-ae74-4d88-b916-0c7cca7a8a58.cfargotunnel.com` | Proxied |
| CNAME | `www` | `af2decc7-ae74-4d88-b916-0c7cca7a8a58.cfargotunnel.com` | Proxied |

Email/SPF records (leave untouched): `_dmarc`, `_domainkey`, `_domainconnect`, SPF TXT.

## Python environment

- **Python 3.11** required — Ubuntu 22.04 ships with 3.10, which is incompatible with `networkx==3.6.1` in requirements.txt
- Venv at `/home/ubuntu/legal_rag/venv`
- Installed from `requirements.txt` (committed to GitHub, generated from local Mac venv)

```bash
sudo apt install -y python3.11 python3.11-venv python3.11-dev
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ollama models

```bash
ollama pull nomic-embed-text   # query embeddings (768-dim)
ollama pull llama3.2           # LLM synthesis
```

## Known OCI gotcha — iptables REJECT rule

Ubuntu on OCI ships with a pre-installed iptables `REJECT all` rule (rule 5 in INPUT chain) that blocks all ports except 22, silently overriding UFW. This was fixed:

```bash
sudo iptables -D INPUT 5
sudo netfilter-persistent save
```

`iptables-persistent` is installed and the fix is persisted across reboots. If port 8501 ever stops being reachable, check `sudo iptables -L INPUT -n --line-numbers` for a `REJECT all` rule and remove it again.

OCI Security List also has an explicit ingress rule: TCP port 8501 from `0.0.0.0/0`.

UFW rules:
```
8501/tcp    ALLOW
OpenSSH     ALLOW
```

## Deploying updates

```bash
ssh ubuntu@163.192.97.229
cd legal_rag
git pull
sudo systemctl restart audit-app
```

If `requirements.txt` changed:
```bash
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart audit-app
```

## BM25 cache

On first startup after a fresh install, BM25 fetches all chunks from Supabase and builds an on-disk index. This takes several minutes and only happens once — subsequent startups load from cache in ~30 seconds. All lawyer sessions share the same in-memory index. Cache lives in `data_files/bm25_cache/` on the server.

## Admin workflow

1. Lawyer visits `https://ejp-rag-audit.com` → clicks **Sign Up**
2. Account created with `approved = false` in `lawyer_accounts` Supabase table
3. You log into admin panel (Login screen → **Admin** tab → enter `ADMIN_PASSWORD`)
4. Approve the account → lawyer can now log in
5. Email the lawyer manually to let them know they're approved

## Cost and expiry

OCI trial credit: ~$300, burns at ~$50/month → exhausted around Oct/Nov 2026. After that, either:
- Pay ~$50/month to continue on Oracle
- Migrate to Hetzner CX32 (~$10/month, 4 vCPU, 8 GB RAM) — setup is identical, just a new server

Domain auto-renews at Squarespace for ~$14/year. If the project winds down, cancel before renewal.
