import re
import matplotlib.pyplot as plt

# Path to the Stage II training log
log_file = "vnet_4ch_bce_kl1136653.log"

# Regex patterns
epoch_pattern = re.compile(r"Epoch\s+(\d+)/(\d+)")
train_loss_pattern = re.compile(r"Train Loss:\s*([\d\.]+)")
val_loss_pattern = re.compile(r"val loss:\s*([\d\.]+)")
train_ed_pattern = re.compile(r"train ED mm:\s*([\d\.]+)")
val_ed_pattern = re.compile(r"val ED mm:\s*([\d\.]+)")
early_stop_pattern = re.compile(r"Early stopping triggered at epoch\s+(\d+)")

# Containers
epochs, train_loss, val_loss, train_ed, val_ed = [], [], [], [], []
early_stop_epoch = None
current_epoch = None

# Parse the log file
with open(log_file, "r") as f:
    for line in f:
        if line.strip().startswith("Saved new best model"):
            continue

        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epochs.append(current_epoch)

        train_match = train_loss_pattern.search(line)
        if train_match:
            train_loss.append(float(train_match.group(1)))

        val_match = val_loss_pattern.search(line)
        if val_match:
            val_loss.append(float(val_match.group(1)))

        train_ed_match = train_ed_pattern.search(line)
        if train_ed_match:
            train_ed.append(float(train_ed_match.group(1)))

        val_ed_match = val_ed_pattern.search(line)
        if val_ed_match:
            val_ed.append(float(val_ed_match.group(1)))

        es_match = early_stop_pattern.search(line)
        if es_match:
            early_stop_epoch = int(es_match.group(1))

plt.figure(figsize=(10, 6))
plt.plot(epochs[:len(train_loss)], train_loss, label="Train Loss", color="blue")
plt.plot(epochs[:len(val_loss)], val_loss, label="Val Loss", color="green")
if early_stop_epoch:
    plt.axvline(x=early_stop_epoch, color="red", linestyle="--", label=f"Early Stopping (Epoch {early_stop_epoch})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Stage II: Training and Validation Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("stage2_loss_curves.png", dpi=300)

# --- Plot ED Curves ---
plt.figure(figsize=(10, 6))
plt.plot(epochs[:len(train_ed)], train_ed, label="Train ED (mm)", color="blue")
plt.plot(epochs[:len(val_ed)], val_ed, label="Val ED (mm)", color="green")
if early_stop_epoch:
    plt.axvline(x=early_stop_epoch, color="red", linestyle="--", label=f"Early Stopping (Epoch {early_stop_epoch})")
plt.xlabel("Epoch")
plt.ylabel("Euclidean Distance (mm)")
plt.title("Stage II: Training and Validation Euclidean Distance")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("stage2_ed_curves.png", dpi=300)
print(train_ed,val_ed,len(train_ed),len(val_ed))
print("✅ Done: Saved loss and ED plots")
