# Celebrity Face Similarity Search with CLIP and MongoDB
This project is a small end-to-end system that takes an input face image and finds the most visually similar celebrity face from a dataset.
It uses CLIP embeddings to represent images as vectors and MongoDB to store and search those embeddings.

The idea for this project came from a previous notebook where I detected faces from a group photo(generated with gpt) using OpenCV and Haar cascades. Then cropped each face, computed CLIP embeddings, and stored them in MongoDB and then compared a new face image to the stored faces using cosine similarity.

In this new project I reuse the same embedding and similarity ideas, but apply them to a real celebrity dataset instead of synthetic images.


## Dataset
The dataset is from Kaggle:

**Celebrity Face Image Dataset**
[https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset)

It contains 18 Hollywood celebrities, each with 100 images:

Angelina Jolie, Brad Pitt, Denzel Washington, Hugh Jackman, Jennifer Lawrence, Johnny Depp, Kate Winslet, Leonardo DiCaprio, Megan Fox, Natalie Portman, Nicole Kidman, Robert Downey Jr., Sandra Bullock, Scarlett Johansson, Tom Cruise, Tom Hanks, Will Smith.

I do not include the dataset in the repository.
To run the notebook, download it from Kaggle and place it in a local folder (for example: `Celebrity Faces Dataset/`).

Each subfolder corresponds to one celebrity and contains that person’s images. This folder structure is important because I use the folder name as the celebrity label.



The main idea is:

1. Use **CLIP** to convert each image into a numerical vector (an embedding).
2. Store these embeddings and their metadata (celebrity name, filename, path) in **MongoDB**.
3. For a new input image:

   * Compute its CLIP embedding.
   * Compare it with all stored embeddings using **cosine similarity**.
   * Rank celebrities by similarity and show the most similar match (and the top 5 results).

No training is required. CLIP is a pre-trained model that already knows a lot about visual features, so I only need to use it as a feature extractor.


## Why I used these tools
### CLIP (image encoder)

* CLIP is a general-purpose vision–language model that produces meaningful image embeddings.
* It works well out of the box, without fine-tuning.
* Compared to training a custom face recognition model, this approach is simpler and faster for a personal project.
* Embeddings can be reused for many tasks (similarity search, clustering, visualization) without changing the pipeline.

Model used:
`openai/clip-vit-base-patch32` from the `transformers` library.

### MongoDB (storage)

* I wanted a simple way to:

  * Persist embeddings between runs.
  * Query and inspect them easily.
* MongoDB stores each image as a document containing:

  * `filename`
  * `celebrity`
  * `image_path`
  * `embedding` (a list of floats)
* A document database is convenient for experiments, debugging, and extending metadata later.

### Cosine similarity (comparison)

* Cosine similarity measures the angle between two vectors and is commonly used for embedding spaces.
* It is scale-invariant, which is useful when the magnitude of embeddings is less important than their direction.
* It is easy to implement with NumPy and works well for ranking similarity in this context.


### Imports and setup

I import:

* `os`, `csv` for file and metadata handling
* `pymongo` and `MongoClient` for MongoDB
* `PIL.Image` for image loading
* `numpy` for array operations
* `torch` for model execution
* `transformers` (`CLIPProcessor`, `CLIPModel`) for the CLIP encoder
* `IPython.display` for displaying images in the notebook

These are the core dependencies needed to run the embedding and similarity pipeline.


### Connect to MongoDB

```python
client = MongoClient("mongodb://localhost:27017/")
db = client.face_db
collection = db.celebrities
```

I create a database called `face_db` and a collection called `celebrities`.
All embeddings and metadata for the celebrity images are stored in this collection.


### Load CLIP encoder

```python
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
```

The processor handles resizing, normalization and tensor conversion.
The model outputs the image features that I use as embeddings.


### Embedding function

```python
def compute_embedding(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.squeeze().numpy().tolist()
```

Step by step:

1. Load the image and convert it to RGB (CLIP expects RGB input).
2. Use the CLIP processor to preprocess it.
3. Use `torch.no_grad()` because there is no training, only inference.
4. Extract the image features and convert the tensor to a NumPy array, then to a Python list so it can be stored in MongoDB.


### Path to dataset

```python
dataset_folder = "Celebrity Faces Dataset"
csv_filename = "celebrity_dataset_metadata.csv"
```

`dataset_folder` must point to the root directory of the Kaggle dataset (where each celebrity has its own subfolder).

### Create metadata CSV and populate MongoDB

I loop through the dataset folder like this:

1. For each celebrity folder:

   * Skip non-folders.
   * Print the name for debugging.

2. For each image file in that folder:

   * Skip non-image files by checking extensions.
   * Build the full `image_path`.
   * Write a row to a CSV file containing:

     * `filename`
     * `celebrity`
     * `image_path`
   * Compute the embedding with `compute_embedding(image_path)`.
   * Insert or update a document in MongoDB:

```python
collection.update_one(
    {"filename": filename},
    {
        "$set": {
            "filename": filename,
            "celebrity": celebrity_name,
            "image_path": image_path,
            "embedding": embedding,
        }
    },
    upsert=True
)
```

The CSV file is useful because:

* It gives a quick global view of the dataset.
* It can be opened in Excel or other tools without touching MongoDB.
* It acts as a simple backup of file paths and labels.

MongoDB keeps both metadata and embeddings in a format that can be queried programmatically.


## Query: find the most similar celebrity to an input image

This part of the notebook uses the stored embeddings to find the closest celebrity match.

### Connect to MongoDB again

For the query phase I reconnect to the same database and collection:

```python
client = MongoClient("mongodb://localhost:27017/")
db = client.face_db
collection = db.celebrities
```

### Load CLIP encoder again

I reload the same CLIP model and processor to ensure the embeddings for the query image are compatible with the stored embeddings.

### Embedding function for the query

The same `compute_embedding` function is reused.

### Cosine similarity

```python
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

This function returns a value between -1 and 1, where 1 means the vectors are identical in direction.
Higher values indicate more similar images.

### Load the query image

```python
query_image = "generatedImage4.png"
query_emb = compute_embedding(query_image)
```

`generatedImage4.png` is an example image.
To test another image, I only need to change the filename here (for example, a photo of myself or another generated face).

### Fetch all celebrity embeddings

```python
docs = list(collection.find({}))
```

This retrieves all stored documents from MongoDB, one per celebrity image.

### Compute similarity and rank results

```python
results = []

for doc in docs:
    score = cosine_similarity(query_emb, doc["embedding"])
    results.append({
        "celebrity": doc["celebrity"],
        "filename": doc["filename"],
        "image_path": doc["image_path"],
        "score": score
    })

results = sorted(results, key=lambda x: x["score"], reverse=True)
best = results[0]
```

This loop:

1. Computes cosine similarity between the query embedding and every celebrity embedding.
2. Stores the relevant information in a list.
3. Sorts the list by similarity score in descending order.
4. Selects the best match as the first element.

### Show the final result and top-k matches

I print:

* The most similar celebrity name.
* The similarity score.
* The filename of the matched image.

Then I display the matched image using `IPython.display.Image`.

There is also a section that prints the **top 5 matches**:

```python
print("\nTop 5 Matches:")
for r in results[:5]:
    print(r["celebrity"], "-", r["score"])
```

This gives a better idea of how the model ranks similar faces.


## How to run the project

**Install requirements**

```bash
pip install pymongo pillow numpy torch torchvision torchaudio transformers
```

**Install and run MongoDB locally**

Make sure MongoDB is running on `mongodb://localhost:27017/`.
The notebook assumes this URI.

**Download the dataset**

Download the “Celebrity Face Image Dataset” from Kaggle and extract it.
Set the `dataset_folder` variable in the notebook to the correct path.

**Run the notebook**

Run all cells in order:

* First section: build embeddings, create CSV, and populate MongoDB.
* Second section: choose a query image and run the similarity search.

**Try different input images**

Replace `generatedImage4.png` with another image path and re-run the query cells.


## Relation to the previous project

The previous notebook (faceRecognition):

* Used OpenCV and a Haar cascade to detect faces inside a group photo.
* Cropped each detected face and saved them in a `stored-faces/` folder.
* Used CLIP to compute embeddings for each cropped face and saved them in MongoDB.
* Compared a new input face with all stored faces using cosine similarity to find the most similar one.

This celebrity project is a natural extension of that idea:

* Instead of first detecting faces in one picture, I use a curated dataset of faces.
* The same CLIP + MongoDB + cosine similarity pattern is reused.
* The logic is more general and can be adapted to other image similarity tasks.

---

## Possible extensions

Some future improvements that could be added:

* Face detection and automatic cropping for any arbitrary input photo.
* Web interface where users can upload a picture and see their top matching celebrities.
* Normalizing embeddings and storing them as NumPy arrays on disk for faster search.
* Using approximate nearest-neighbor search libraries for scalability when the dataset grows.

---

This notebook is mainly a learning and portfolio project to explore image embeddings, database storage, and similarity search in a clear, step-by-step way.