"""
summary_datasize.csv -> gain_vs_baseline vs r (aug only) 꺾은선 그래프.
x축: r (log2 스케일), y축: gain_vs_baseline, 색: n_key 별.
"""
import pandas as pd
import matplotlib.pyplot as plt

CSV = "summary_datasize.csv"       # 같은 폴더에 두고 실행
OUT = "gain_vs_r.png"

df = pd.read_csv(CSV)

# aug 조건만, r>0 (baseline r=0은 gain 정의상 0/빈칸)
aug = df[(df["kind"] == "aug") & (df["r"] > 0)].copy()

# n_key를 데이터 크기 순으로 정렬
order = sorted(aug["n_key"].unique(), key=lambda k: int(k[1:]))

fig, ax = plt.subplots(figsize=(7, 5))
for nk in order:
    sub = aug[aug["n_key"] == nk].sort_values("r")
    ax.plot(sub["r"], sub["gain_vs_baseline"] * 100,   # %p 로 표기
            marker="o", label=nk)

ax.set_xscale("log", base=2)
ax.set_xticks([0.5, 1, 2, 4, 8])
ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
ax.axhline(0, color="gray", lw=0.8, ls="--")           # 이득 0 기준선
ax.set_xlabel("Augmentation ratio  r  (synthetic = r × real, log2)")
ax.set_ylabel("Accuracy gain vs baseline (%p)")
ax.set_title("Diffusion augmentation gain by real-data size (aug)")
ax.legend(title="real data size")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved:", OUT)
