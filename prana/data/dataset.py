

from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset



def load_prana_dataset(
    repo_id: str = "Siddarth09/PRANA",
    root: str | None = None,
):
    """
    Load PRANA dataset using LeRobotDataset (correct way).
    """
    ds = LeRobotDataset(
        repo_id=repo_id,
        root=root,             
        download_videos=False,  # set True only if you need images now
    )
    return ds


def inspect_dataset(ds: LeRobotDataset):
    print("\n=== BASIC INFO ===")
    print(ds)

    print("\n=== FEATURES ===")
    for k, v in ds.features.items():
        print(f"{k}: dtype={v['dtype']} shape={v.get('shape')}")

    print("\n=== CAMERA KEYS ===")
    print(ds.meta.camera_keys)

    print("\n=== STATS KEYS ===")
    for k in ds.meta.stats.keys():
        print(k)

    print("\n=== EPISODES ===")
    print("Total episodes:", ds.meta.total_episodes)
    print("Total frames:", ds.meta.total_frames)

    lengths = ds.meta.episodes["length"]
    print("Min episode length:", min(lengths))
    print("Max episode length:", max(lengths))


if __name__ == "__main__":
    ds = load_prana_dataset()
    inspect_dataset(ds)
