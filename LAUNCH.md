# ragpilot — Launch Readiness

**Date:** 2026-08-27
**Owner:** Conrad CJ Wilson
**Repo:** cjps4linux-creator/ragpilot
**Status:** Functional reference implementation

---

## Readiness Snapshot

| Gate | Status | Evidence |
|---|---|---|
| CI passing | Pending | GitHub Actions workflow defined |
| Tests passing | Pending | pytest suite in `backend/tests/` |
| Security scan | Pending | SECURITY.md in place |
| README complete | Complete | Architecture, JD mapping, honest limitations |
| LICENSE | Complete | MIT — Conrad CJ Wilson |
| Docker build | Complete | Dockerfile, docker-compose.yml, Procfile present |
| Deployment config | Complete | railway.toml for one-click PaaS deploy |

---

## Requirements

- Python 3.11+
- OpenAI API key OR AWS credentials with Bedrock access (for production mode)
- No external dependencies in mock mode

---

## Known Gaps

- CI has not been verified running on GitHub Actions for this commit
- Golden eval faithfulness uses lexical overlap; NLI scorer is a future improvement
- Knowledge-graph relationships are co-occurrence only; typed relationships are a future improvement
- No authentication layer; deploy behind API gateway or reverse proxy for production

---

## Actions Required Before Production

1. Verify CI passes on GitHub Actions
2. Run golden eval in production mode (real embeddings + LLM)
3. Enable GitHub secret scanning and vulnerability alerts
4. Configure branch protection with required status checks
5. Add authentication layer if exposing publicly

---

## Contact

Conrad CJ Wilson — conradcjwilson0@gmail.com
