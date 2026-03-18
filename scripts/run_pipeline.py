import subprocess
import sys

cmd = [
    sys.executable,
    "-m",
    "orchestration.pipeline",
    "run",
    "--split", "train",
    "--preprocess",
    "--use_gt_masks",
    "--extract_lesions",
    "--train_embed",
    "--embed_all",
    "--build_neo4j",
    "--run_guarded_rag",
]

print("======================================")
print(" Running Capstone End-to-End Pipeline ")
print("======================================")

subprocess.check_call(cmd)

print("\n✅ Full pipeline finished successfully.")