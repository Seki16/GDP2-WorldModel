import csv
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    csv_path = Path("evaluation/baseline_metrics.csv")
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    steps = []
    ep_returns = []
    ep_lens = []
    epsilons = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            ep_returns.append(float(row["episode_return"]))
            ep_lens.append(int(row["episode_len"]))
            epsilons.append(float(row["epsilon"]))

    def moving_avg(x, n=20):
        out = []
        for i in range(len(x)):
            j0 = max(0, i - n + 1)
            out.append(sum(x[j0:i+1]) / (i - j0 + 1))
        return out

    ma = moving_avg(ep_returns, n=20)

    plt.figure()
    plt.plot(range(len(ep_returns)), ep_returns, label="Episode return")
    plt.plot(range(len(ma)), ma, label="Moving avg (20 eps)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DQN Baseline on MazeEnv")
    plt.legend()
    out_img = Path("evaluation/baseline_return_curve.png")
    plt.savefig(out_img, dpi=200, bbox_inches="tight")
    plt.close()

    last_k = min(50, len(ep_returns))
    avg_last = sum(ep_returns[-last_k:]) / last_k
    best = max(ep_returns) if ep_returns else float("nan")
    avg_len_last = sum(ep_lens[-last_k:]) / last_k

    summary = Path("evaluation/baseline_summary.txt")
    summary.write_text(
        "\n".join([
            "DQN Baseline Summary (MazeEnv)",
            f"- Episodes logged: {len(ep_returns)}",
            f"- Best episode return: {best:.4f}",
            f"- Avg return (last {last_k} eps): {avg_last:.4f}",
            f"- Avg episode length (last {last_k} eps): {avg_len_last:.2f}",
            "",
            "Files produced:",
            "- evaluation/baseline_return_curve.png",
            "- evaluation/baseline_summary.txt",
        ]),
        encoding="utf-8"
    )

    print(f"Saved: {out_img}")
    print(f"Saved: {summary}")


if __name__ == "__main__":
    main()
