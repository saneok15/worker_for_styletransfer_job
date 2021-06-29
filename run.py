import pickle
import asyncio
import app
import numpy as np
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import matplotlib
import websockets
import pymongo
import gridfs
from PIL import Image
from tqdm import tqdm

from app.funcs import load_image, im_convert, get_features, gram_matrix

with open('p/mongo') as f:
    mongo_login = f.readline()
    print(mongo_login)
client = pymongo.MongoClient(mongo_login)
print(client.test)
db = client['diplom_images']
fs = gridfs.GridFS(db)

vgg= torch.load('models/vgg19features')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)

style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}

content_weight = 1  
style_weight = 1e6  

show_every = 200

async def predict(websocket, path):

    img_ids = await websocket.recv()
    img_ids_s = pickle.loads(img_ids)
    print('start_func'.join(str(x) for x in img_ids_s))

    content_mongo_im = fs.get(img_ids_s[0])
    style_mongo_im = fs.get(img_ids_s[1])

    content = load_image(content_mongo_im).to(device)
    style = load_image(style_mongo_im, shape=content.shape[-2:]).to(device)

    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = content.clone().requires_grad_(True).to(device)

    optimizer = optim.Adam([target], lr=0.006)
    steps = 1000
    content_loss_lst = []
    for ii in tqdm(range(1, steps + 1)):

        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        content_loss_lst.append(content_loss)

        style_loss = 0
        
        for layer in style_weights:
            
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape

            
            target_gram = gram_matrix(target_feature)

            style_gram = style_grams[layer]
            
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)

            
            style_loss += layer_style_loss / (d * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print('predict end')
    result_filename = 'imgs/'+str(img_ids_s[0])+'_'+str(img_ids_s[1])+'.png'
    
    plt.imsave(result_filename, im_convert(target))
    try:
        with open(result_filename, 'rb') as f:
            im = f.read()
    except:
        print('open failed')

    try:
        target = fs.put(im, filename=result_filename)
        print('put')
    except:
        print('put gone wrong')

    try:
        result_mongo_id = fs.get(target )._id
        print(result_mongo_id)
    except:
        print('smth gone wrong')


    await websocket.send(pickle.dumps([result_mongo_id]))
    print(f"> {result_mongo_id}")


start_server = websockets.serve(predict, 'localhost', 8765)
print('ok')

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()