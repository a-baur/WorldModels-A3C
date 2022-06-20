import numpy as np
import os, sys, glob
import gym
from hparams import HyperParams as hp
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
import cv2

def rollout():
    env = gym.make("BipedalWalker-v3")

    # load pretrained ppo agent
    checkpoint = load_from_hub(repo_id="mrm8488/ppo-BipedalWalker-v3", filename="bipedalwalker-v3.zip")
    model = PPO.load(checkpoint)

    # Resize images from (600 x 400) to (180x120)
    scale_percent = 30 # percent of original size
    width = int(600 * scale_percent / 100)
    height = int(400 * scale_percent / 100)
    dim = (width, height)


    seq_len = 2000
    max_ep = hp.n_rollout
    feat_dir = hp.data_dir

    os.makedirs(feat_dir, exist_ok=True)

    for ep in range(max_ep):
        env.seed(ep) # every episode has different seed
        #obs_lst, action_lst, reward_lst, next_obs_lst, done_lst = [], [], [], [], []
        obs = env.reset()
        img = env.render(mode = "rgb_array") # initial obs
        # Downscale image
        img_downscaled = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        done = False
        t = 0
        
        while not done and t < seq_len:
            t += 1

            #use pretrained model to predict actions
            action, _state = model.predict(obs)

            #action = env.action_space.sample()  # sample random action

            next_obs, reward, done, _ = env.step(action)

            next_img = env.render(mode = "rgb_array")

            # Downscale next image
            next_img_downscaled = cv2.resize(next_img, dim, interpolation = cv2.INTER_AREA)


            # np.savez(
            #     os.path.join(feat_dir, 'rollout_{:03d}_{:04d}'.format(ep,t)),
            #     obs=img_downscaled, # save img as observations
            #     action=action,
            #     reward=reward,
            #     next_obs=next_img_downscaled,
            #     done=done,
            # )
            
            obs_lst.append(img_downscaled)
            action_lst.append(action)
            reward_lst.append(reward)
            next_obs_lst.append(next_img_downscaled)
            done_lst.append(done)

            obs = next_obs
            img_downscaled = next_img_downscaled

        np.savez(
            os.path.join(feat_dir, 'rollout_ep_{:03d}'.format(ep)),
            obs=np.stack(obs_lst, axis=0), # (T, C, H, W)
            action=np.stack(action_lst, axis=0), # (T, a)
            reward=np.stack(reward_lst, axis=0), # (T, 1)
            next_obs=np.stack(next_obs_lst, axis=0), # (T, C, H, W)
            done=np.stack(done_lst, axis=0), # (T, 1)
        )
        
        

if __name__ == '__main__':
    np.random.seed(hp.seed)
    rollout()
