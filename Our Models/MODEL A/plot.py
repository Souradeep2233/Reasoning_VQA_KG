import re
import matplotlib.pyplot as plt
from datetime import datetime

def parse_log_file(filepath):
    # Data containers
    train_data = {
        'updates': [],
        'loss': [],
        'lr': [],
        'ups': [],
        'max_mem': []
    }
    
    val_data = {
        'updates': [],
        'loss': [],
        'accuracy': []
    }

    # Regex patterns for extraction
    # Matches: "progress: 100/88000" and extracts 100
    update_pattern = re.compile(r"progress:\s+(\d+)/")
    
    # Training patterns
    train_loss_pattern = re.compile(r"train/total_loss:\s+([\d\.]+)")
    lr_pattern = re.compile(r"lr:\s+([\d\.]+)")
    ups_pattern = re.compile(r"ups:\s+([\d\.]+)")
    mem_pattern = re.compile(r"max mem:\s+([\d\.]+)")
    
    # Validation patterns
    val_loss_pattern = re.compile(r"val/total_loss:\s+([\d\.]+)")
    val_acc_pattern = re.compile(r"val/okvqa/vqa_accuracy:\s+([\d\.]+)")

    try:
        with open(filepath, 'r') as f:
            for line in f:
                # We only care about log lines from the logistics callback
                if "mmf.trainers.callbacks.logistics" not in line:
                    continue

                # Extract the update step (x-axis)
                update_match = update_pattern.search(line)
                if not update_match:
                    continue
                
                step = int(update_match.group(1))

                # Check if it is a Training Log
                if "train/total_loss" in line:
                    loss_match = train_loss_pattern.search(line)
                    lr_match = lr_pattern.search(line)
                    ups_match = ups_pattern.search(line)
                    mem_match = mem_pattern.search(line)

                    if loss_match:
                        train_data['updates'].append(step)
                        train_data['loss'].append(float(loss_match.group(1)))
                        
                        # Optional fields (might not be in every line)
                        train_data['lr'].append(float(lr_match.group(1)) if lr_match else 0)
                        train_data['ups'].append(float(ups_match.group(1)) if ups_match else 0)
                        train_data['max_mem'].append(float(mem_match.group(1)) if mem_match else 0)

                # Check if it is a Validation Log
                elif "val/total_loss" in line:
                    loss_match = val_loss_pattern.search(line)
                    acc_match = val_acc_pattern.search(line)

                    if loss_match and acc_match:
                        val_data['updates'].append(step)
                        val_data['loss'].append(float(loss_match.group(1)))
                        val_data['accuracy'].append(float(acc_match.group(1)))

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None, None

    return train_data, val_data

def plot_metrics(train_data, val_data):
    if not train_data or not val_data:
        return

    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Analysis', fontsize=16)

    # 1. Loss Curve (Train vs Val)
    axs[0, 0].plot(train_data['updates'], train_data['loss'], label='Train Loss', color='tab:blue', alpha=0.7)
    axs[0, 0].plot(val_data['updates'], val_data['loss'], label='Val Loss', color='tab:red', marker='o', linestyle='--')
    axs[0, 0].set_title('Loss over Time')
    axs[0, 0].set_xlabel('Updates')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Validation Accuracy
    axs[0, 1].plot(val_data['updates'], val_data['accuracy'], label='Val Accuracy', color='tab:green', marker='o')
    axs[0, 1].set_title('Validation Accuracy')
    axs[0, 1].set_xlabel('Updates')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Learning Rate
    axs[1, 0].plot(train_data['updates'], train_data['lr'], label='Learning Rate', color='tab:purple')
    axs[1, 0].set_title('Learning Rate Schedule')
    axs[1, 0].set_xlabel('Updates')
    axs[1, 0].set_ylabel('LR')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. Training Speed (UPS) & Memory
    ax4 = axs[1, 1]
    ax4.plot(train_data['updates'], train_data['ups'], label='UPS', color='tab:orange')
    ax4.set_xlabel('Updates')
    ax4.set_ylabel('Updates/Sec', color='tab:orange')
    ax4.tick_params(axis='y', labelcolor='tab:orange')
    
    # Create a second y-axis for Memory
    ax4_mem = ax4.twinx()
    # Filter out 0 memory values for plotting cleanly
    mem_updates = [u for u, m in zip(train_data['updates'], train_data['max_mem']) if m > 0]
    mem_values = [m for m in train_data['max_mem'] if m > 0]
    
    ax4_mem.plot(mem_updates, mem_values, label='Max Memory', color='tab:gray', linestyle=':', alpha=0.5)
    ax4_mem.set_ylabel('Max Memory (MB)', color='tab:gray')
    ax4_mem.tick_params(axis='y', labelcolor='tab:gray')
    
    axs[1, 1].set_title('System Performance (Speed & Memory)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    log_file = "train.log" 
    print(f"Parsing {log_file}...")
    t_data, v_data = parse_log_file(log_file)
    
    if t_data and v_data:
        print(f"Found {len(t_data['updates'])} training points and {len(v_data['updates'])} validation points.")
        plot_metrics(t_data, v_data)
    else:
        print("No data found to plot.")