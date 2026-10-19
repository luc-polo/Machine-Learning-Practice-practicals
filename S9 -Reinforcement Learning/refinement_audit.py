import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

NOTEBOOK = next(Path(__file__).resolve().parent.glob('*.ipynb'))
AUDIT_DIR = NOTEBOOK.parent / '.cache' / 'q_refinement'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
notebook = json.loads(NOTEBOOK.read_text())
exec(''.join(next(c for c in notebook['cells'] if c['id'] == 'c009')['source'])
     .split('for name, env in')[0])
exec(''.join(next(c for c in notebook['cells'] if c['id'] == 'c016')['source'])
     .split('def plot_rewards')[0])

GAMMAS = np.round(np.arange(.974, .9921, .0005), 4)
ALPHAS = [.1, .3]
SEEDS = [101, 211, 307]
EPISODES = 100_000


class SharedSampleQAgent:
    """Each table gets one ordinary Q-learning update per observed transition."""
    def __init__(self, env, seed):
        self.gammas = np.repeat(GAMMAS, len(ALPHAS))
        self.alphas = np.tile(ALPHAS, len(GAMMAS))
        self.tables = np.zeros((len(self.gammas), env.observation_space.n,
                                env.action_space.n))
        self.visits = np.zeros(self.tables.shape[1:], dtype=np.int64)
        index = np.flatnonzero((self.gammas == .98) & (self.alphas == .3))[0]
        self.q = self.tables[index]
        self.rng = np.random.default_rng(seed)
        self.epsilon = 1.
        self.episode = 0
        self.total_q = np.zeros_like(self.tables)
        self.snapshots = 0
        self.started = time.perf_counter()

    def act(self, s):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q.shape[1]))
        values = self.q[s]
        best = np.flatnonzero(values == values.max())
        return int(best[0] if len(best) == 1 else self.rng.choice(best))

    def train(self, s, a, r, s2, done):
        self.visits[s, a] += 1
        rates = self.alphas / (1 + self.visits[s, a] / 100) ** .6
        targets = (np.full(len(self.gammas), r) if done else
                   r + self.gammas * self.tables[:, s2, :].max(axis=1))
        self.tables[:, s, a] += rates * (targets - self.tables[:, s, a])

    def end_episode(self):
        self.epsilon = max(.05, .9999 * self.epsilon)
        self.episode += 1
        if self.episode >= 70_000 and self.episode % 100 == 0:
            self.total_q += self.tables
            self.snapshots += 1
        if self.episode % 20_000 == 0:
            print(f'  {self.episode:,} episodes; '
                  f'{time.perf_counter()-self.started:.1f}s', flush=True)


def train_all():
    for seed in SEEDS:
        target = AUDIT_DIR / f'run_{seed}.npz'
        if target.exists():
            print('Already trained', seed, flush=True)
            continue
        print('Training seed', seed, flush=True)
        env = gym.make('FrozenLake-v1', map_name='8x8')
        env.reset(seed=seed)
        agent = SharedSampleQAgent(env, seed)
        rewards = run_training(env, agent, EPISODES)
        seconds = time.perf_counter() - agent.started
        np.savez_compressed(target, q=agent.tables,
                            average_q=agent.total_q / agent.snapshots,
                            gammas=agent.gammas, alphas=agent.alphas,
                            rewards=rewards, seconds=seconds)
        env.close()


def validate():
    saved = [np.load(AUDIT_DIR / f'run_{seed}.npz') for seed in SEEDS]
    raw = np.array([d['q'] for d in saved]).mean(axis=0)
    tail = np.array([d['average_q'] for d in saved]).mean(axis=0)
    candidates = {}
    for i, (gamma, alpha) in enumerate(zip(saved[0]['gammas'], saved[0]['alphas'])):
        for averaging, tables in [('final',raw),('tail',tail)]:
            candidates[f'g={gamma}, alpha={alpha}, {averaging}'] = tables[i]
    candidates['previous ensemble'] = np.load(NOTEBOOK.parent/'q_table.npy')
    scores_by_policy = {}
    scores = {}
    env = gym.make('FrozenLake-v1', map_name='8x8')
    for name, q in candidates.items():
        policy = q.argmax(axis=1)
        key = policy.astype(np.uint8).tobytes()
        if key not in scores_by_policy:
            scores_by_policy[key] = evaluate(env, policy, n=5000, seed=300_000)
        scores[name] = scores_by_policy[key]
    for name, score in sorted(scores.items(), key=lambda item:-item[1])[:15]:
        print(name, f'{score:.4f}', flush=True)
    print('Previous ensemble:', scores['previous ensemble'],
          '; distinct policies:', len(scores_by_policy), flush=True)
    best = max(scores, key=scores.get)
    np.savez_compressed(AUDIT_DIR / 'selection.npz', q=candidates[best],
                        selected=best, score=scores[best],
                        scores_json=json.dumps(scores), n=5000, seed=300000)
    env.close()




def test_frozen_challenger():
    with np.load(AUDIT_DIR / 'selection.npz',allow_pickle=False) as selected:
        challenger=selected['q'].copy()
        selection=str(selected['selected'])
        validation_scores=json.loads(str(selected['scores_json']))
    prior=np.load(NOTEBOOK.parent/'q_table.npy',allow_pickle=False)
    with np.load(NOTEBOOK.parent/'training_runs.npz',allow_pickle=False) as saved:
        baseline=saved['baseline_q'][0].copy()

    def episode_results(q, n=20_000, seed=400_000):
        env=gym.make('FrozenLake-v1',map_name='8x8')
        policy=q.argmax(axis=1)
        wins=np.zeros(n)
        for i in range(n):
            state,_=env.reset(seed=seed+i)
            done=False
            while not done:
                state,r,terminated,truncated,_=env.step(int(policy[state]))
                wins[i]+=r
                done=terminated or truncated
        env.close()
        return wins

    arrays={}
    for name,q in [('original',baseline),('previous_ensemble',prior),('refinement',challenger)]:
        arrays[name]=episode_results(q)
        print(name,arrays[name].mean(),flush=True)

    differences={}
    for name in ['original','previous_ensemble']:
        delta=arrays['refinement']-arrays[name]
        mean=float(delta.mean())
        half=1.959963984540054*float(delta.std(ddof=1))/np.sqrt(len(delta))
        differences[name]=dict(gain=mean,paired_95ci=[mean-half,mean+half])
        print('gain versus',name,mean,'paired 95% CI',mean-half,mean+half,flush=True)

    report=dict(
        config=dict(gymnasium=gym.__version__,environment='FrozenLake-v1',map_name='8x8',
                    max_episode_steps=100,gammas=GAMMAS.tolist(),alphas=ALPHAS,
                    seeds=SEEDS,episodes_per_seed=EPISODES,
                    behaviour_gamma=.98,behaviour_alpha=.3,
                    epsilon_decay=.9999,epsilon_min=.05,alpha_schedule='alpha/(1+visits/100)**0.6',
                    tail_average_start=70000,tail_average_frequency=100,
                    notes='Each table receives one model-free Q-learning update per shared sampled transition.'),
        selected=selection,
        validation=dict(n=5000,seed=300000,candidates=validation_scores,
                        previous=validation_scores['previous ensemble'],
                        best=max(validation_scores.values())),
        fresh_test=dict(n=20000,seed=400000,scores={name:float(a.mean()) for name,a in arrays.items()},
                        improvements=differences),
        seconds=float(sum(np.load(AUDIT_DIR / f'run_{seed}.npz')['seconds'] for seed in SEEDS)))
    (NOTEBOOK.parent / 'refinement_report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    train_all()
    validate()
    test_frozen_challenger()
