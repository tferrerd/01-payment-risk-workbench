"""Rehearse release restoration with exact local images in an isolated container."""
import argparse
from datetime import datetime, timezone
import json
import subprocess
import uuid


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


def drill(previous, candidate):
    images = [json.loads(docker("image", "inspect", image))[0] for image in [previous, candidate]]
    assert images[0]["Id"] != images[1]["Id"], "Drill needs two different image IDs"
    name = "payment-risk-rollback-" + uuid.uuid4().hex[:10]
    phases = []
    for phase, index in [("previous_baseline", 0), ("candidate", 1), ("rollback", 0), ("restore_candidate", 1)]:
        image = images[index]
        manifest = "deployment/explorer-v2.json" if index else "deployment/explorer-v1.json"
        docker("run", "--detach", "--name", name, "--read-only", "--tmpfs", "/tmp",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", image["Id"])
        try:
            evidence = json.loads(docker("exec", name, "python", "scripts/check_release.py",
                                        "--manifest", manifest, "--url", "http://127.0.0.1:8000"))
            assert evidence["saved_prediction_parity"] and evidence["explanation_reconstructs_score"]
            phases.append({"phase": phase, "image_id": image["Id"],
                           "architecture": image["Architecture"], "checks": evidence})
        finally:
            docker("rm", "--force", name)
    assert phases[0]["image_id"] == phases[2]["image_id"]
    assert phases[1]["image_id"] == phases[3]["image_id"]
    assert len({item["checks"]["artifact_sha256"] for item in phases}) == 1
    return {"checked_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Isolated local Docker application-release rollback; same model in both releases. Not a Render rollback or a model-version rollback.",
            "phases": phases, "previous_image_restored": True, "candidate_restored_after_drill": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-image", required=True)
    parser.add_argument("--candidate-image", required=True)
    args = parser.parse_args()
    print(json.dumps(drill(args.previous_image, args.candidate_image), indent=2))


if __name__ == "__main__":
    main()
