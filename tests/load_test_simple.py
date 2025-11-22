# tests/perf/load_test_simple.py(performance)
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER = "http://127.0.0.1:5000"
CONCURRENCY = 20
REQUESTS = 100
TIMEOUT = 10

def make_request(session, text):
    try:
        # login first as admin to get session cookie
        session.post(SERVER + "/login", data={"username":"admin","password":"admin"}, timeout=TIMEOUT)
        resp = session.post(SERVER + "/dashboard", data={"text": text}, timeout=TIMEOUT)
        return resp.status_code, len(resp.content)
    except Exception as e:
        return 0, str(e)

def main():
    text = "This is a sample document. " * 200
    start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = []
        for i in range(REQUESTS):
            s = requests.Session()
            futures.append(ex.submit(make_request, s, text))
        for f in as_completed(futures):
            results.append(f.result())
    elapsed = time.perf_counter() - start
    ok = sum(1 for r in results if r[0] == 200)
    print(f"Requests: {REQUESTS}, Success: {ok}, Elapsed: {elapsed:.2f}s, RPS: {REQUESTS/elapsed:.2f}")

if __name__ == "__main__":
    main()
