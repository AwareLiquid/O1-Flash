import io

p = "benchmarks/real_data_bench.py"
s = io.open(p, encoding="utf-8").read()
s = s.replace('proxies = {"http": "http://127.0.0.1:9674",\n                   "https": "http://127.0.0.1:9674"}',
              'proxies = None')
s = s.replace("r = requests.get(_BASE + name, proxies=proxies, timeout=120)",
              "r = requests.get(_BASE + name, timeout=120)")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched")
