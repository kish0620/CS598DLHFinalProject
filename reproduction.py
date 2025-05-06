import torch
from pyhealth.datasets import MIMIC3Dataset
# from pyhealth.tasks import Task
from torch.utils.data import DataLoader, Dataset
import datetime

# 1. Load the MIMIC-III dataset
mimic3_ds = MIMIC3Dataset(
    root="physionet.org/files/mimiciii/1.4",  # path where your mimic3 data is stored
    tables=[]
)
# print(mimic3_ds.patients)
# # 2. Filter patients aged 18+ on 2020-12-03
# def filter_adults(patient_data):
#     # Get DOB from the PATIENTS table (assuming patient_data is a dictionary or similar structure)
#     print(patient_data)
#     dob = patient_data['DOB']  # Format: YYYY-MM-DD
#     dob = datetime.datetime.strptime(dob, "%Y-%m-%d")

#     # Reference date
#     ref_date = datetime.datetime(2020, 12, 3)
    
#     # Compute age
#     age = (ref_date - dob).days / 365.25
#     return age >= 18

# # 3. Apply the filter manually to the dataset
# filtered_patients = []
# for patient_data in mimic3_ds.patients:  # Assuming mimic3_ds.PATIENTS contains the patient data
#     if filter_adults(patient_data):
#         filtered_patients.append(patient_data)


# Assuming you are using a dataset object with a method to create tasks, like MIMIC3Dataset
# We define a custom task using the dataset (this part depends on your dataset API)
# You can adapt this to the specific API of your dataset class
# Define a simple dataset for the mortality prediction task
class MortalityDataset(Dataset):
    def __init__(self, dataset, label_fn):
        self.dataset = dataset  # dict: patient_id -> Patient object
        self.label_fn = label_fn

        # For indexing and ordering
        self.data_list = list(self.dataset.values())

        # Infer input/output dimensions from first patient/visit
        first_patient = self.data_list[0]
        first_visit = next(iter(first_patient.visits.values()))  # get first visit object

        # Example feature: number of events (currently 0 in your case)
        self.input_dim = 1  # Placeholder; adapt as needed
        self.output_dim = 2  # Binary classification

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        patient = self.data_list[idx]
        visit = next(iter(patient.visits.values()))
        label = self.label_fn(visit)

        features = torch.tensor([[0.0]], dtype=torch.float32)  # <-- notice [[ ]] for 2D!
        label = torch.tensor(label, dtype=torch.long)
        length = torch.tensor(1, dtype=torch.long)  # dummy length for 1 visit

        return {
            "input": features,
            "label": label.item(),
            "length": length,
        }



# Mortality label function based on discharge status
def mortality_label_fn(visit):
    return int(visit.discharge_status)

# 1. Prepare your data (Assuming mimic3_ds.patients has the patient data)
patients = mimic3_ds.patients  # This will be your list of patient data
# print(patients)

# 2. Create the dataset for mortality prediction
mortality_dataset = MortalityDataset(patients, mortality_label_fn)

# 3. Split the dataset into train, validation, and test sets
train_size = int(0.8 * len(mortality_dataset))  # 80% for training
test_size = len(mortality_dataset) - train_size # Remaining for testing

input_dim = mortality_dataset.input_dim
trainset, testset = torch.utils.data.random_split(
    mortality_dataset, [train_size, test_size]
)

# 4. Create DataLoaders
train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
test_loader = DataLoader(testset, batch_size=32, shuffle=False)

print('Preprocessing done')

import torch.nn as nn
import torch.optim as optim

# Define the RNN model
class RNNMortalityModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.1):
        super(RNNMortalityModel, self).__init__()
        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, lengths):
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed)
        out = self.fc(hidden[-1])  # Use last hidden layer
        return out.squeeze(1)
    
class TransformerMortalityModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, num_heads=8, dropout=0.1):
        super(TransformerMortalityModel, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)  # Embedding layer for input features
        
        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dropout=dropout
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers
        )
        
        # Linear layer to predict the output
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, lengths):
        # Reshape input: [batch_size, seq_len, input_dim] -> [seq_len, batch_size, input_dim]
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, input_dim]
        
        # Apply the embedding layer
        x = self.embedding(x)
        
        # Pass through transformer
        transformer_output = self.transformer_encoder(x)
        
        # We only care about the last token for the final prediction
        out = transformer_output[-1, :, :]  # Get last time-step (for sequence-to-label)
        
        # Pass through the fully connected layer to get the prediction
        out = self.fc(out)
        
        return out.squeeze(1)
    
# from sklearn.linear_model import SGDClassifier

# def compute_cav(model, concept_loader, random_loader):
#     model.eval()

#     def get_activations(loader):
#         activations = []
#         for batch in loader:
#             inputs, lengths = batch['input'], batch['length']
#             with torch.no_grad():
#                 acts = model(inputs, lengths, return_activations=True)
#             activations.append(acts.cpu())
#         return torch.cat(activations).numpy()

#     concept_acts = get_activations(concept_loader)
#     random_acts = get_activations(random_loader)

#     X = np.concatenate([concept_acts, random_acts], axis=0)
#     y = np.array([1] * len(concept_acts) + [0] * len(random_acts))

#     cav_model = SGDClassifier(alpha=0.01, max_iter=1000)
#     cav_model.fit(X, y)

#     cav = cav_model.coef_[0]  # CAV is the weight vector
#     return cav

# def compute_tcav_score(model, loader, cav):
#     model.eval()
#     directional_derivatives = []

#     for batch in loader:
#         inputs, lengths = batch['input'], batch['length']
#         inputs.requires_grad = True

#         # Forward pass
#         acts = model(inputs, lengths, return_activations=True)
#         output = model.fc(acts)  # apply final layer manually

#         # Compute directional derivative
#         for i in range(output.shape[0]):
#             model.zero_grad()
#             output[i].backward(retain_graph=True)

#             grad = inputs.grad[i][-1]  # only last time step
#             dot = torch.dot(torch.from_numpy(cav).float(), grad)
#             directional_derivatives.append(dot.item())

#     tcav_score = np.mean([d > 0 for d in directional_derivatives])
#     return tcav_score



# Model, loss, optimizer
model = RNNMortalityModel(input_dim=input_dim)
# Extension
# model = TransformerMortalityModel(input_dim=input_dim)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Training loop
for epoch in range(1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs, labels, lengths = batch['input'], batch['label'], batch['length']

        inputs, labels = inputs, labels.float()

        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}")

import torch
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input']
            labels = batch['label']
            lengths = batch['length']

            outputs = model(inputs, lengths)  # model forward
            probs = torch.sigmoid(outputs).squeeze()  # assuming binary classification

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Finally calculate metrics
    from sklearn.metrics import roc_auc_score, average_precision_score

    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)

    print(f"Test AUROC: {auroc:.4f}")
    print(f"Test AUPRC: {auprc:.4f}")


evaluate(model, test_loader)




