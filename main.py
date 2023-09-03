import gymnasium as gym

from stable_baselines3 import DQN
from env import example_env, gcn_env
from greatx.training import Trainer
from greatx.nn.models import GCN, SGC
from torch_geometric.datasets import KarateClub
from greatx.utils import split_nodes
from greatx.attack.injection import AdvInjection, RandomInjection
from greatx.nn.models import RobustGCN, MedianGCN

# env = gym.make("CartPole-v1", render_mode="human")
# model = DQN("MlpPolicy", env, verbose=1)
# env = example_env.CustomEnv(size=5)
# model = DQN(
#     policy="MultiInputPolicy",
#     env=env,
#     verbose=1,
#     device="cuda",
# )

dataset = KarateClub()
data = dataset[0]
num_features = data.x.size(-1)
num_classes = data.y.max().item() + 1
splits = split_nodes(data.y, random_state=100, train=0.3, test=0.5, val=0.2)


for atk_model_name in ["RandomInjection", "AdvInjection"]:
    for gnn_model_name in ["GCN", "SGC"]:
        print(f"{atk_model_name}_{gnn_model_name}\n")
        if atk_model_name == "AdvInjection":
            atk_model = AdvInjection
        elif atk_model_name == "RandomInjection":
            atk_model = RandomInjection

        if gnn_model_name == "GCN":
            gnn_model = GCN
        elif gnn_model_name == "SGC":
            gnn_model = SGC

        # Before
        trainer_before = Trainer(gnn_model(num_features, num_classes), device="cuda")
        trainer_before.fit(data, mask=(splits.train_nodes, splits.val_nodes))
        logs = trainer_before.evaluate(data, splits.test_nodes)
        acc_before = logs["acc"]

        # Attack
        attacker = atk_model(data, device="cuda")
        if atk_model_name == "AdvInjection":
            attacker.setup_surrogate(trainer_before.model)
        attacker.reset()
        if atk_model_name == "AdvInjection":
            attacker.attack(3, feat_budgets=10)
        else:
            attacker.attack(3, feat_limits=(0, 1))
        logs = trainer_before.evaluate(attacker.data(), splits.test_nodes)
        acc_after = logs["acc"]

        # Defense compare 1
        compare1_model = RobustGCN(num_features, num_classes)
        compare1_trainer = Trainer(compare1_model, device="cuda")
        compare1_trainer.fit(
            attacker.data(), mask=(splits.train_nodes, splits.val_nodes)
        )
        logs = compare1_trainer.evaluate(attacker.data(), splits.test_nodes)
        acc_compare1 = logs["acc"]

        # Defense compare 2
        compare2_model = MedianGCN(num_features, num_classes, hids=[], acts=[])
        compare2_trainer = Trainer(compare2_model, device="cuda")
        compare2_trainer.fit(
            attacker.data(), mask=(splits.train_nodes, splits.val_nodes)
        )
        logs = compare2_trainer.evaluate(attacker.data(), splits.test_nodes)
        acc_compare2 = logs["acc"]

        # with open("/home/miku/reinforce-gcn/result.txt", "a") as file:
        #     file.write(f"{atk_model_name}_{gnn_model_name}\n")
        #     file.write(
        #         f"CLN: {acc_before:.3};\t After {atk_model_name:.3} ATK: {acc_after:.3};\tRobustGCN:{acc_compare1:.3};\tMedianGCN:{acc_compare2:.3}\n"
        #     )

        # Defense
        env = gcn_env.GcnEnv(data=attacker.data(), splits=splits, max_b=20)
        model = DQN(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            device="cuda",
        )

        model.learn(total_timesteps=50, log_interval=4)
        model.save(f"{gnn_model_name}_{atk_model_name}_karateclub")

        obs, info = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                with open("/home/miku/reinforce-gcn/result.txt", "a") as file:
                    file.write(f"{atk_model_name}_{gnn_model_name}\n")
                    file.write(
                        f"CLN: {acc_before:.3};\t After {atk_model_name:.3} ATK: {acc_after:.3};\tOurs: {info['acc']:.3};\tRobustGCN:{acc_compare1:.3};\tMedianGCN:{acc_compare2:.3}\n"
                    )
                print(
                    f"CLN: {acc_before:.3};\t After {atk_model_name:.3} ATK: {acc_after:.3};\tOurs: {info['acc']:.3};\tRobustGCN:{acc_compare1:.3};\tMedianGCN:{acc_compare2:.3}\n"
                )
                obs, info = env.reset()
                break
