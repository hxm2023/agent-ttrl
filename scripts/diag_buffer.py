import sys
sys.path.insert(0, "src")
from agent_ttrl.optimization.replay_buffer import EvidenceRow, ReplayBuffer

rb = ReplayBuffer(capacity=256, anchor_fraction=0.4)
# simulate: pos rows and neg rows across intents
for i in range(40):
    rb.add(EvidenceRow(f"p{i}", "F1_refund", [i], [i + 1], advantage=1.0, policy_version=0))
    rb.add(EvidenceRow(f"n{i}", "F1_refund", [i], [i + 1], advantage=-1.0, policy_version=0))
    rb.add(EvidenceRow(f"px{i}", "F1_exchange", [i], [i + 1], advantage=1.0, policy_version=0))
    rb.add(EvidenceRow(f"nx{i}", "F1_exchange", [i], [i + 1], advantage=-1.0, policy_version=0))
batch = rb.sample_update_batch(48, seed=0)
pos = [r for r in batch if r.advantage > 0]
neg = [r for r in batch if r.advantage < 0]
print("batch size:", len(batch), "pos:", len(pos), "neg:", len(neg))
pairs = list(zip(pos[:24], neg[:24]))
print("pairs:", len(pairs))
