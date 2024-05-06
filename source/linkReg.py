
# Import necessary libraries
import pandas as pd
import numpy as np
import torch
# import os

from torch_geometric.data import download_url, extract_zip
import pandas as pd

def main():
    
    # os.environ['TORCH'] = torch.__version__
    # !pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}.html
    # !pip install git+https://github.com/pyg-team/pytorch_geometric.git
    
    # !pip install sentence_transformers
    # !pip3 install fuzzywuzzy[speedup]
    # !pip install captum


    
    dataset_name = 'ml-latest-small'
    
    url = f'https://files.grouplens.org/datasets/movielens/{dataset_name}.zip'
    extract_zip(download_url(url, '.'), '.')
    
    movies_path = f'./{dataset_name}/movies.csv'
    ratings_path = f'./{dataset_name}/ratings.csv'

    # Load the entire ratings dataframe into memory:
    ratings_df = pd.read_csv(ratings_path)[["userId", "movieId", "rating"]]
    
    # Load the entire movie dataframe into memory:
    movies_df = pd.read_csv(movies_path, index_col='movieId')
    
    print('movies.csv:')
    print('===========')
    print(movies_df[["genres", "title"]].head())
    print(f"Number of movies: {len(movies_df)}")
    print()
    print('ratings.csv:')
    print('============')
    print(ratings_df[["userId", "movieId", "rating"]].head())
    print(f"Number of ratings: {len(ratings_df)}")
    print()

    from fuzzywuzzy import fuzz
    
    # Specify your userId
    our_user_id = ratings_df['userId'].max() + 1
    
    print('Most rated movies:')
    print('==================')
    most_rated_movies = ratings_df['movieId'].value_counts().head(10)
    print(movies_df.loc[most_rated_movies.index][["title"]])
    
    # Initialize your rating list
    ratings = []

    # Add your ratings here:
    num_ratings = 5
    # while len(ratings) < num_ratings:
    #     print(f'Select the {len(ratings) + 1}. movie:')
    #     print('=====================================')
    #     movie = input('Please enter the movie title: ')
    #     movies_df['title_score'] = movies_df['title'].apply(lambda x: fuzz.ratio(x, movie))
    #     print(movies_df.sort_values('title_score', ascending=False)[['title']].head(5))
    #     movie_id = input('Please enter the movie id: ')
    #     if not movie_id:
    #         continue
    #     movie_id = int(movie_id)
    #     rating = float(input('Please enter your rating: '))
    #     if not rating:
    #         continue
    #     assert 0 <= rating <= 5
    #     ratings.append({'movieId': movie_id, 'rating': rating, 'userId': our_user_id})
    #     print()

    # Add your ratings to the rating dataframe
    # ratings_df = pd.concat([ratings_df, pd.DataFrame.from_records(ratings)])

    # # Select our userId
    # our_user_id = ratings_df['userId'].max() + 1
    
    # # Load the IMDB ratings:
    # imdb_rating_path = f'./imdb_ratings.csv'
    # imdb_ratings_df = pd.read_csv(imdb_rating_path)
    # imdb_ratings_df.columns = imdb_ratings_df.columns.str.strip().str.lower()
    
    # # The IMDB movie titles / ids do not match the movie titles /ids in the movielens dataframes
    # # so we need to map them:
    # imdb_ratings_df['title'] = imdb_ratings_df['title'] + ' (' + imdb_ratings_df['year'].astype(str) + ')'
    # imdb_ratings_df['title'] = imdb_ratings_df['title'].str.strip()
    # movies_df['title'] = movies_df['title'].str.strip()
    # imdb_ratings_df = imdb_ratings_df.merge(movies_df['title'].reset_index(), on='title', how='inner', )
    
    # # The ratings are on a scale from 1 to 10, so we need to transform them to a scale from 0 to 5:
    # imdb_ratings_df['rating'] = (imdb_ratings_df['your rating'] / 2).astype(int)
    
    # # Your ratings that we are going to use:
    # print('Your IMDB ratings:')
    # print('==================')
    # print(imdb_ratings_df[['title', 'rating']].head(10))
    
    # # Finally, we can add the ratings to the ratings data frame:
    # imdb_ratings_df['userId'] = our_user_id
    # ratings_df = pd.concat([ratings_df, imdb_ratings_df[['movieId', 'rating', 'userId']]])

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    
    # One-hot encode the genres:
    genres = movies_df['genres'].str.get_dummies('|').values
    genres = torch.from_numpy(genres).to(torch.float)
    
    # Load the pre-trained sentence transformer model and encode the movie titles:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with torch.no_grad():
        titles = model.encode(movies_df['title'].tolist(), convert_to_tensor=True, show_progress_bar=True)
        titles = titles.cpu()
    
    # Concatenate the genres and title features:
    movie_features = torch.cat([genres, titles], dim=-1)
    
    # We don't have user features, which is why we use an identity matrix
    user_features = torch.eye(len(ratings_df['userId'].unique()))


    # Create a mapping from the userId to a unique consecutive value in the range [0, num_users]:
    unique_user_id = ratings_df['userId'].unique()
    unique_user_id = pd.DataFrame(data={
        'userId': unique_user_id,
        'mappedUserId': pd.RangeIndex(len(unique_user_id))
        })
    print("Mapping of user IDs to consecutive values:")
    print("==========================================")
    print(unique_user_id.head())
    print()
    
    # Create a mapping from the movieId to a unique consecutive value in the range [0, num_movies]:
    unique_movie_id = ratings_df['movieId'].unique()
    unique_movie_id = pd.DataFrame(data={
        'movieId': unique_movie_id,
        'mappedMovieId': pd.RangeIndex(len(unique_movie_id))
        })
    print("Mapping of movie IDs to consecutive values:")
    print("===========================================")
    print(unique_movie_id.head())
    print()
    
    # Merge the mappings with the original data frame:
    ratings_df = ratings_df.merge(unique_user_id, on='userId')
    ratings_df = ratings_df.merge(unique_movie_id, on='movieId')
    
    # With this, we are ready to create the edge_index representation in COO format
    # following the PyTorch Geometric semantics:
    edge_index = torch.stack([
        torch.tensor(ratings_df['mappedUserId'].values),
        torch.tensor(ratings_df['mappedMovieId'].values)]
        , dim=0)
    
    assert edge_index.shape == (2, len(ratings_df))
    
    print("Final edge indices pointing from users to movies:")
    print("================================================")
    print(edge_index[:, :10])

    import torch_geometric.transforms as T
    from torch_geometric.data import HeteroData
    
    # Create the heterogeneous graph data object:
    data = HeteroData()
    
    # Add the user nodes:
    data['user'].x = user_features  # [num_users, num_features_users]
    
    # Add the movie nodes:
    data['movie'].x = movie_features  # [num_movies, num_features_movies]
    
    # Add the rating edges:
    data['user', 'rates', 'movie'].edge_index = edge_index  # [2, num_ratings]
    
    # Add the rating labels:
    rating = torch.from_numpy(ratings_df['rating'].values).to(torch.float)
    data['user', 'rates', 'movie'].edge_label = rating  # [num_ratings]
    
    # We also need to make sure to add the reverse edges from movies to users
    # in order to let a GNN be able to pass messages in both directions.
    # We can leverage the `T.ToUndirected()` transform for this from PyG:
    data = T.ToUndirected()(data)
    
    # With the above transformation we also got reversed labels for the edges.
    # We are going to remove them:
    del data['movie', 'rev_rates', 'user'].edge_label
    
    assert data['user'].num_nodes == len(unique_user_id)
    assert data['user', 'rates', 'movie'].num_edges == len(ratings_df)
    assert data['movie'].num_features == 404
    
    print(data)

    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=0.0,
        edge_types=[('user', 'rates', 'movie')],
        rev_edge_types=[('movie', 'rev_rates', 'user')],
    )(data)
    print(train_data)
    print(val_data)
    print(test_data)

    from torch_geometric.nn import SAGEConv, to_hetero
    
    class GNNEncoder(torch.nn.Module):
        def __init__(self, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), hidden_channels)
            self.conv2 = SAGEConv((-1, -1), out_channels)
    
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x
    
    
    class EdgeDecoder(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
            self.lin2 = torch.nn.Linear(hidden_channels, 1)
    
        def forward(self, z_dict, edge_label_index):
            row, col = edge_label_index
            z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)
    
            z = self.lin1(z).relu()
            z = self.lin2(z)
            return z.view(-1)
    
    
    class Model(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            self.encoder = GNNEncoder(hidden_channels, hidden_channels)
            self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
            self.decoder = EdgeDecoder(hidden_channels)
    
        def forward(self, x_dict, edge_index_dict, edge_label_index):
            z_dict = self.encoder(x_dict, edge_index_dict)
            return self.decoder(z_dict, edge_label_index)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Model(hidden_channels=32).to(device)
    
    print(model)

    import torch.nn.functional as F
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    def train():
        model.train()
        optimizer.zero_grad()
        pred = model(train_data.x_dict, train_data.edge_index_dict,
                     train_data['user', 'movie'].edge_label_index)
        target = train_data['user', 'movie'].edge_label
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        return float(loss)
    
    @torch.no_grad()
    def test(data):
        data = data.to(device)
        model.eval()
        pred = model(data.x_dict, data.edge_index_dict,
                     data['user', 'movie'].edge_label_index)
        pred = pred.clamp(min=0, max=5)
        target = data['user', 'movie'].edge_label.float()
        rmse = F.mse_loss(pred, target).sqrt()
        return float(rmse)
    
    
    for epoch in range(1, 301):
        train_data = train_data.to(device)
        loss = train()
        train_rmse = test(train_data)
        val_rmse = test(val_data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
              f'Val: {val_rmse:.4f}')

    with torch.no_grad():
        test_data = test_data.to(device)
        pred = model(test_data.x_dict, test_data.edge_index_dict,
                     test_data['user', 'movie'].edge_label_index)
        pred = pred.clamp(min=0, max=5)
        target = test_data['user', 'movie'].edge_label.float()
        rmse = F.mse_loss(pred, target).sqrt()
        print(f'Test RMSE: {rmse:.4f}')
    
    userId = test_data['user', 'movie'].edge_label_index[0].cpu().numpy()
    movieId = test_data['user', 'movie'].edge_label_index[1].cpu().numpy()
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    print(pd.DataFrame({'userId': userId, 'movieId': movieId, 'rating': pred, 'target': target}))

#            userId  movieId    rating  target
# 0         468     1515  3.726183     3.0
# 1         134       39  3.255185     2.0
# 2         413     3251  3.234478     3.0
# 3         425      481  3.222397     4.5
# 4         598     1149  3.518827     2.5
# ...       ...      ...       ...     ...
# 10078     553      131  3.319815     4.0
# 10079     306       73  3.218582     3.0
# 10080     200     1455  4.566834     4.0
# 10081     367     1513  2.757144     3.0
# 10082     598     1145  2.843732     2.5

    # Your mappedUserId
    mapped_user_id = unique_user_id[unique_user_id['userId'] == our_user_id]['mappedUserId'].values[0]
    
    # Select movies that you haven't seen before
    movies_rated = ratings_df[ratings_df['mappedUserId'] == mapped_user_id]
    movies_not_rated = movies_df[~movies_df.index.isin(movies_rated['movieId'])]
    movies_not_rated = movies_not_rated.merge(unique_movie_id, on='movieId')
    movie = movies_not_rated.sample(1)
    
    print(f"The movie we want to predict a raiting for is:  {movie['title'].item()}")

    # Create new `edge_label_index` between the user and the movie
    edge_label_index = torch.tensor([
        mapped_user_id,
        movie.mappedMovieId.item()])
    
    
    with torch.no_grad():
        test_data.to(device)
        pred = model(test_data.x_dict, test_data.edge_index_dict, edge_label_index)
        pred = pred.clamp(min=0, max=5).detach().cpu().numpy()

    pred.item()

    from torch_geometric.explain import Explainer, CaptumExplainer
    
    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model',
        model_config=dict(
            mode='regression',
            task_level='edge',
            return_type='raw',
        ),
        node_mask_type=None,
        edge_mask_type='object',
    )
    
    explanation = explainer(
        test_data.x_dict, test_data.edge_index_dict, index=0,
        edge_label_index=edge_label_index).cpu().detach()
    explanation

    # User to movie link + attribution
    user_to_movie = explanation['user', 'movie'].edge_index.numpy().T
    user_to_movie_attr = explanation['user', 'movie'].edge_mask.numpy().T
    user_to_movie_df = pd.DataFrame(
        np.hstack([user_to_movie, user_to_movie_attr.reshape(-1,1)]),
        columns = ['mappedUserId', 'mappedMovieId', 'attr']
    )
    
    # Movie to user link + attribution
    movie_to_user = explanation['movie', 'user'].edge_index.numpy().T
    movie_to_user_attr = explanation[ 'movie', 'user'].edge_mask.numpy().T
    movie_to_user_df = pd.DataFrame(
        np.hstack([movie_to_user, movie_to_user_attr.reshape(-1,1)]),
        columns = ['mappedMovieId', 'mappedUserId','attr']
    )
    explanation_df = pd.concat([user_to_movie_df, movie_to_user_df])
    explanation_df[["mappedUserId", "mappedMovieId"]] = explanation_df[["mappedUserId", "mappedMovieId"]].astype(int)
    
    print(f"Attribtion for all edges towards prediction of movie rating of movie:\n {movie['title'].item()}")
    print("==========================================================================================")
    print(explanation_df.sort_values(by='attr'))

    # Select links that connect to our user
    explanation_df = explanation_df[explanation_df['mappedUserId'] == mapped_user_id]
    
    # We group the attribution scores by movie
    explanation_df = explanation_df.groupby('mappedMovieId').sum()
    
    # Merge with movies_df to receive title
    # But first, we need to add the original id
    explanation_df = explanation_df.merge(unique_movie_id, on='mappedMovieId')
    explanation_df = explanation_df.merge(movies_df, on='movieId')
    
    pd.options.display.float_format = "{:,.9f}".format
    
    print("Top movies that influenced the prediction:")
    print("==============================================")
    print(explanation_df.sort_values(by='attr', ascending=False, key= lambda x: abs(x))[['title', 'attr']].head())


if __name__ == "__main__":
    main()