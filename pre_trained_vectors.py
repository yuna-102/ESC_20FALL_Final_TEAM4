from tqdm import tqdm_notebook
import gensim
import zipfile

def load_pretrained_fasttext(word2idx, fname):
    """Load pretrained vectors and create embedding layers.
    
    Args:
        word2idx (Dict): Vocabulary built from the corpus
        fname (str): Path to pretrained vector file

    Returns:
        embeddings (np.array): Embedding matrix with shape (N, d) where N is
            the size of word2idx and d is embedding dimension
    """

    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())

    # Initilize random embeddings
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
    embeddings[word2idx['<pad>']] = np.zeros((d,))

    # Load pretrained vectors
    count = 0
    for line in tqdm_notebook(fin):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in word2idx:
            count += 1
            embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

    print(f"There are {count} / {len(word2idx)} pretrained vectors found.")

    return embeddings


def load_pretrained_word2vec(word2idx, fname):
    """Load pretrained vectors and create embedding layers.
    
    Args:
        word2idx (Dict): Vocabulary built from the corpus
        fname (str): Path to pretrained vector file

    Returns:
        embeddings (np.array): Embedding matrix with shape (N, d) where N is
            the size of word2idx and d is embedding dimension
    """

    print("Loading pretrained vectors...")
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)  
    n = word2vec_model.vectors.shape[0]
    d = word2vec_model.vectors.shape[1]
    

    # Initilize random embeddings
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
    embeddings[word2idx['<pad>']] = np.zeros((d,))

    def get_vector(word):
      if word in word2vec_model:
          return word2vec_model[word]
      else:
          return None

    # Load pretrained vectors
    count = 0
    for word, i in word2idx.items():
        temp = get_vector(word) 
        if temp is not None:
          count += 1
          embeddings[word2idx[word]] = temp

    print(f"There are {count} / {len(word2idx)} pretrained vectors found.")

    return embeddings


def get_vector(vectors):
    vectors = args.
    print("Loading pretrained vectors...")
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
    target_path = 'crawl-300d-2M.vec.zip'
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())

    with zipfile.ZipFile('crawl-300d-2M.vec.zip', 'r') as zip_ref:
      zip_ref.extractall()
