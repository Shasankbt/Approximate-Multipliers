from binary_semantic_classifier.tokenizer import Tokenizer
from binary_semantic_classifier.dataset import IMDB_Dataset
from torch.utils.data import DataLoader
import torch
from binary_semantic_classifier.model import SemanticClassifier


MAX_LENGTH = 512
BATCH_SIZE = 32
MODEL_PTH = "binary_semantic_classifier/semantic_classifier_model.pth"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

tokenizer = Tokenizer()
val_dataset = IMDB_Dataset(data_path="binary_semantic_classifier/imdb_val.csv", max_length=MAX_LENGTH, tokenizer=tokenizer)

val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
criterion = torch.nn.CrossEntropyLoss()

model = SemanticClassifier(
    num_classes=2,
    vocab_size=tokenizer.get_vocab_size(),
    max_len=MAX_LENGTH
).to(device)

model.load_state_dict(torch.load(MODEL_PTH, map_location=device))


def run_evaluation():
    model.eval()

    total_val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, labels in val_dataloader:
            print(total)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            break

    avg_val_loss = total_val_loss / len(val_dataloader)
    accuracy = 100 * correct / total

    print(f"\n{'='*60}")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  Val Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_evaluation()