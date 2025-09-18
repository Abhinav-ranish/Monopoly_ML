
# Advanced Monopoly RL with DQN, richer rules, scripted opponents, and logging.
# Key upgrades vs previous:
# - Fuller rules: Chance/Chest events, Jail (go to jail, get out by rolling doubles or paying), Utilities & Railroads,
#   auctions when a player declines to buy, simplified mortgaging (auto-mortgage lowest-value props if cash < 0).
# - Trading: multi-item (k-for-m) plus optional cash sweetener; +$100 premium for completing sets for the receiver;
#   stochastic "lean to accept" behavior near fair; safety checks.
# - Agent: DQN (MLP) with target network; epsilon decay; experience replay.
# - Opponents: toggle scripted bots (aggressive buyer, conservative, builder) vs. self-play pool.
# - Evaluation: landing distribution, win rates, average wealth; CSV logs and weights checkpoints.
# - CLI flags for episodes, eval-every, opponent type, save-dir.
#
# Usage:
#   python monopoly_rl_advanced.py --episodes 5000 --eval-every 250 --opponents scripted --save-dir ./mono_adv
#
# NOTE: This is a compact educational simulator, not a full rules-accurate engine.
import argparse, os, random, math, json, csv, time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

# ============== Utils ==============
random.seed(123)
np.random.seed(123)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def roll_two():
    d1 = random.randint(1, 6)
    d2 = random.randint(1, 6)
    return d1 + d2, (d1 == d2), (d1, d2)

# ============== Board Data (full 40 tiles; simplified rents) ==============
# Tile types: GO, PROPERTY, RR, UTILITY, TAX, CHANCE, CHEST, JAIL, GOTOJAIL, FREEPARK
GO = "GO"
PROP = "PROP"
RR = "RR"
UTIL = "UTIL"
TAX = "TAX"
CHANCE = "CHANCE"
CHEST = "CHEST"
JAIL = "JAIL"
GOTOJAIL = "GOTOJAIL"
FREEPARK = "FREEPARK"

@dataclass
class Tile:
    name: str
    ttype: str
    color: Optional[str] = None
    price: int = 0
    base_rent: int = 0
    house_cost: int = 0

# Simplified but complete board list (index 0..39)
def build_board():
    tiles = [None]*40
    tiles[0]  = Tile("GO", GO)
    tiles[1]  = Tile("Mediterranean", PROP, "brown", 60, 2, 50)
    tiles[2]  = Tile("Community Chest", CHEST)
    tiles[3]  = Tile("Baltic", PROP, "brown", 60, 4, 50)
    tiles[4]  = Tile("Income Tax", TAX, price=0, base_rent=200)
    tiles[5]  = Tile("Reading RR", RR, "rail", 200, 25)
    tiles[6]  = Tile("Oriental", PROP, "lightblue", 100, 6, 50)
    tiles[7]  = Tile("Chance", CHANCE)
    tiles[8]  = Tile("Vermont", PROP, "lightblue", 100, 6, 50)
    tiles[9]  = Tile("Connecticut", PROP, "lightblue", 120, 8, 50)
    tiles[10] = Tile("Jail / Just Visiting", JAIL)
    tiles[11] = Tile("St. Charles", PROP, "purple", 140, 10, 100)
    tiles[12] = Tile("Electric Company", UTIL, "util", 150, 4)
    tiles[13] = Tile("States", PROP, "purple", 140, 10, 100)
    tiles[14] = Tile("Virginia", PROP, "purple", 160, 12, 100)
    tiles[15] = Tile("Penn RR", RR, "rail", 200, 25)
    tiles[16] = Tile("St. James", PROP, "orange", 180, 14, 100)
    tiles[17] = Tile("Community Chest", CHEST)
    tiles[18] = Tile("Tennessee", PROP, "orange", 180, 14, 100)
    tiles[19] = Tile("New York", PROP, "orange", 200, 16, 100)
    tiles[20] = Tile("Free Parking", FREEPARK)
    tiles[21] = Tile("Kentucky", PROP, "red", 220, 18, 150)
    tiles[22] = Tile("Chance", CHANCE)
    tiles[23] = Tile("Indiana", PROP, "red", 220, 18, 150)
    tiles[24] = Tile("Illinois", PROP, "red", 240, 20, 150)
    tiles[25] = Tile("B&O RR", RR, "rail", 200, 25)
    tiles[26] = Tile("Atlantic", PROP, "yellow", 260, 22, 150)
    tiles[27] = Tile("Ventnor", PROP, "yellow", 260, 22, 150)
    tiles[28] = Tile("Water Works", UTIL, "util", 150, 4)
    tiles[29] = Tile("Marvin Gardens", PROP, "yellow", 280, 24, 150)
    tiles[30] = Tile("Go To Jail", GOTOJAIL)
    tiles[31] = Tile("Pacific", PROP, "green", 300, 26, 200)
    tiles[32] = Tile("North Carolina", PROP, "green", 300, 26, 200)
    tiles[33] = Tile("Community Chest", CHEST)
    tiles[34] = Tile("Pennsylvania", PROP, "green", 320, 28, 200)
    tiles[35] = Tile("Short Line", RR, "rail", 200, 25)
    tiles[36] = Tile("Chance", CHANCE)
    tiles[37] = Tile("Park Place", PROP, "blue", 350, 35, 200)
    tiles[38] = Tile("Luxury Tax", TAX, price=0, base_rent=100)
    tiles[39] = Tile("Boardwalk", PROP, "blue", 400, 50, 200)
    return tiles

TILES = build_board()
BOARD_SIZE = 40
GO_BONUS = 200
START_CASH = 1500
MAX_MOVES = 300

# Build color groups
COLOR_GROUPS: Dict[str, List[int]] = {}
for i, t in enumerate(TILES):
    if t is None or t.color is None: continue
    COLOR_GROUPS.setdefault(t.color, []).append(i)
for v in COLOR_GROUPS.values():
    v.sort()

# ============== Chance & Chest (simplified decks) ==============
CHANCE_DECK = [
    ("ADVANCE_GO", None),
    ("GOTO_JAIL", None),
    ("ADVANCE_24", 24),  # Illinois
    ("ADVANCE_11", 11),  # St. Charles
    ("BACK_3", -3),
    ("PAY_BANK", -50),
    ("RECEIVE_BANK", 50),
]
CHEST_DECK = [
    ("RECEIVE_BANK", 200),
    ("RECEIVE_BANK", 50),
    ("PAY_BANK", -50),
    ("GOTO_JAIL", None),
    ("RECEIVE_BANK", 100),
    ("RECEIVE_BANK", 20),
]

def draw_card(deck):
    card = random.choice(deck)
    return card

# ============== Player and Ownership ==============
@dataclass
class Player:
    pid: int
    cash: int = START_CASH
    pos: int = 0
    jail_turns: int = 0
    bankrupt: bool = False
    properties: List[int] = field(default_factory=list)  # indexes
    houses: Dict[int, int] = field(default_factory=dict)  # idx -> 0..4
    mortgaged: Dict[int, bool] = field(default_factory=dict)  # idx -> bool
    get_out_of_jail: int = 0

    def net_worth(self):
        prop_val = sum((0 if self.mortgaged.get(i, False) else TILES[i].price) for i in self.properties)
        house_val = sum(self.houses.get(i,0)*TILES[i].house_cost for i in self.properties if TILES[i].house_cost>0)
        return self.cash + prop_val + house_val

class GameState:
    def __init__(self, n_players=4):
        self.n = n_players
        self.players = [Player(pid=i) for i in range(n_players)]
        self.owner: Dict[int, int] = {}  # idx -> pid
        self.turn = 0
        self.move_count = 0
        self.landing_hist = np.zeros(BOARD_SIZE, dtype=int)

    def alive(self):
        return [p for p in self.players if not p.bankrupt]

    def next_turn(self):
        self.turn = (self.turn + 1) % self.n
        # skip bankrupt
        for _ in range(self.n):
            if not self.players[self.turn].bankrupt:
                break
            self.turn = (self.turn + 1) % self.n

# ============== Trading ==============
def property_value(idx: int, receiver_pid: int, state: GameState) -> int:
    t = TILES[idx]
    v = t.price
    if t.color in COLOR_GROUPS and t.color not in ("rail", "util"):
        group = COLOR_GROUPS[t.color]
        owned_by_receiver = sum(1 for g in group if state.owner.get(g,-1)==receiver_pid)
        if owned_by_receiver == len(group)-1 and state.owner.get(idx, -1) != receiver_pid:
            v += 150  # set completion premium
    return v

def completes_set(idx: int, pid: int, state: GameState) -> bool:
    t = TILES[idx]
    if t.color in ("rail","util", None): return False
    group = COLOR_GROUPS[t.color]
    needed = [g for g in group if state.owner.get(g,-1) != pid]
    return needed == [idx]

def evaluate_trade(give_list: List[int], receive_list: List[int], proposer: Player, other: Player, state: GameState, cash_delta:int=0):
    # Positive cash_delta means proposer pays that cash to other.
    # Value both sides; +$100 premium on items that complete a set for the receiver.
    offer_val = sum(property_value(i, other.pid, state) + (100 if completes_set(i, other.pid, state) else 0) for i in give_list)
    ask_val   = sum(property_value(i, proposer.pid, state) + (100 if completes_set(i, proposer.pid, state) else 0) for i in receive_list)
    # include cash
    offer_val += cash_delta  # proposer giving cash increases value to other
    # Diff for "other": positive means worse for other (they pay more than they receive)
    diff_for_other = ask_val - offer_val
    # Lean accept near fair: sigmoid(-diff/200)
    def accept_prob(d):
        return 1/(1+math.exp(d/200.0))
    other_accepts = (random.random() < accept_prob(diff_for_other))
    # Proposer should also prefer not to be heavily disadvantaged
    diff_for_prop = -diff_for_other
    proposer_accepts = (random.random() < accept_prob(diff_for_prop))
    return other_accepts and proposer_accepts

def execute_trade(give_list, receive_list, proposer: Player, other: Player, state: GameState, cash_delta:int=0):
    for i in give_list:
        if i in proposer.properties:
            proposer.properties.remove(i)
            state.owner[i] = other.pid
            other.properties.append(i)
            proposer.houses.pop(i, None); other.houses.pop(i, None)  # reset houses on traded lots
            proposer.mortgaged.pop(i, None); other.mortgaged[i] = False
    for j in receive_list:
        if j in other.properties:
            other.properties.remove(j)
            state.owner[j] = proposer.pid
            proposer.properties.append(j)
            other.houses.pop(j, None); proposer.houses.pop(j, None)
            other.mortgaged.pop(j, None); proposer.mortgaged[j] = False
    if cash_delta>0 and proposer.cash>=cash_delta:
        proposer.cash -= cash_delta; other.cash += cash_delta

def propose_trade(state: GameState, proposer: Player):
    others = [p for p in state.players if p.pid!=proposer.pid and not p.bankrupt]
    if not others or not proposer.properties: return
    other = random.choice(others)
    if not other.properties: return
    # random small bundles
    give_k = min(len(proposer.properties), random.choice([1,1,2]))
    rec_k  = min(len(other.properties), random.choice([1,1,2]))
    give_list = random.sample(proposer.properties, give_k)
    receive_list = random.sample(other.properties, rec_k)
    cash_delta = random.choice([0, 0, 50, 100])  # proposer pays other sometimes
    if evaluate_trade(give_list, receive_list, proposer, other, state, cash_delta):
        execute_trade(give_list, receive_list, proposer, other, state, cash_delta)

# ============== Rent and Building ==============
def count_owned(pid: int, color: str, state: GameState):
    return sum(1 for i in COLOR_GROUPS[color] if state.owner.get(i,-1)==pid)

def has_full_set(pid: int, color: str, state: GameState):
    return color in COLOR_GROUPS and count_owned(pid, color, state) == len(COLOR_GROUPS[color])

def rr_rent(pid_owner: int, state: GameState):
    # classic: 25, 50, 100, 200 (scaled)
    owned = sum(1 for i in range(40) if TILES[i].ttype==RR and state.owner.get(i,-1)==pid_owner)
    return [0,25,50,100,200][owned]

def util_rent(pid_owner: int, dice_sum: int, state: GameState):
    owned = sum(1 for i in range(40) if TILES[i].ttype==UTIL and state.owner.get(i,-1)==pid_owner)
    mult = 4 if owned==1 else 10
    return mult * dice_sum

def lot_rent(idx: int, state: GameState):
    t = TILES[idx]
    owner = state.owner.get(idx,-1)
    if owner==-1: return 0
    if t.ttype == PROP:
        base = t.base_rent
        houses = state.players[owner].houses.get(idx,0)
        mult = 1 + houses*3
        if has_full_set(owner, t.color, state):
            mult += 1
        return base * mult
    elif t.ttype == RR:
        return rr_rent(owner, state)
    elif t.ttype == UTIL:
        # dice_sum should be passed in; we'll approximate with 7 here (avg)
        return util_rent(owner, 7, state)
    return 0

def can_build(pid: int, state: GameState) -> Optional[int]:
    p = state.players[pid]
    best = None; best_h = 5
    for i in p.properties:
        t = TILES[i]
        if t.ttype != PROP or t.color in ("rail","util"): continue
        if not has_full_set(pid, t.color, state): continue
        h = p.houses.get(i, 0)
        if h < 4 and h < best_h and p.cash >= t.house_cost:
            best = i; best_h = h
    return best

# ============== Taxes, Jail, Chance/Chest, Auctions, Mortgaging ==============
def send_to_jail(p: Player):
    p.pos = 10
    p.jail_turns = 3

def handle_tile(state: GameState, pid: int, dice_sum: int):
    p = state.players[pid]
    tile = TILES[p.pos]
    state.landing_hist[p.pos]+=1

    if tile.ttype == TAX:
        pay = tile.base_rent
        p.cash -= pay
    elif tile.ttype == CHANCE:
        act, val = draw_card(CHANCE_DECK)
        apply_card(state, p, act, val)
    elif tile.ttype == CHEST:
        act, val = draw_card(CHEST_DECK)
        apply_card(state, p, act, val)
    elif tile.ttype == GOTOJAIL:
        send_to_jail(p)
    elif tile.ttype in (PROP, RR, UTIL):
        owner = state.owner.get(p.pos, -1)
        if owner==-1:
            return "UNOWNED"
        elif owner != pid and not state.players[owner].bankrupt:
            # rent
            if tile.ttype == UTIL:
                rent = util_rent(owner, dice_sum, state)
            elif tile.ttype == RR:
                rent = rr_rent(owner, state)
            else:
                rent = lot_rent(p.pos, state)
            transfer_cash(state, pid, owner, rent)
    return None

def transfer_cash(state: GameState, payer_id: int, payee_id: int, amount: int):
    payer = state.players[payer_id]; payee = state.players[payee_id]
    payer.cash -= amount; payee.cash += amount

def auction_property(state: GameState, idx: int, skip_pid: int):
    # Simple: highest random bid among players with enough cash (reserve at 50% list price)
    reserve = TILES[idx].price // 2
    bidders = [p for p in state.players if not p.bankrupt and p.pid!=skip_pid and p.cash>=reserve]
    if not bidders: return
    winner = max(bidders, key=lambda x: random.random()*x.cash)  # noisy preference ~ cash
    price = reserve
    winner.cash -= price
    winner.properties.append(idx)
    state.owner[idx] = winner.pid

def apply_card(state: GameState, p: Player, act: str, val):
    if act == "ADVANCE_GO":
        if p.pos != 0:
            p.cash += GO_BONUS
        p.pos = 0
    elif act == "GOTO_JAIL":
        send_to_jail(p)
    elif act == "ADVANCE_24":
        if 24 < p.pos: p.cash += GO_BONUS
        p.pos = 24
    elif act == "ADVANCE_11":
        if 11 < p.pos: p.cash += GO_BONUS
        p.pos = 11
    elif act == "BACK_3":
        p.pos = (p.pos + val) % BOARD_SIZE
    elif act == "PAY_BANK":
        p.cash += val  # val is negative
    elif act == "RECEIVE_BANK":
        p.cash += val

def auto_mortgage_if_needed(state: GameState, p: Player):
    # If cash < 0, mortgage lowest-price unmortgaged properties until >=0 or bust
    while p.cash < 0:
        candidates = [i for i in p.properties if not p.mortgaged.get(i, False)]
        if not candidates:
            # bankrupt
            p.bankrupt = True
            # release properties
            for i in list(p.properties):
                state.owner.pop(i, None)
            p.properties.clear(); p.houses.clear(); p.mortgaged.clear()
            break
        i = min(candidates, key=lambda x: TILES[x].price)
        p.mortgaged[i] = True
        p.cash += TILES[i].price // 2  # mortgage value

# ============== Agent: DQN ==============
class Replay:
    def __init__(self, cap=50000):
        self.cap = cap
        self.buf = []
        self.pos = 0
    def push(self, *tup):
        if len(self.buf) < self.cap:
            self.buf.append(tup)
        else:
            self.buf[self.pos] = tup
        self.pos = (self.pos+1)%self.cap
    def sample(self, bs):
        idx = np.random.choice(len(self.buf), bs, replace=False)
        return [self.buf[i] for i in idx]
    def __len__(self): return len(self.buf)

def mlp(sizes):
    # simple numpy MLP with ReLU
    W, b = [], []
    for i in range(len(sizes)-1):
        fan_in, fan_out = sizes[i], sizes[i+1]
        w = np.random.randn(fan_in, fan_out) / math.sqrt(fan_in)
        W.append(w); b.append(np.zeros(fan_out))
    return W, b

def forward(x, W, b):
    a = x
    for i in range(len(W)-1):
        z = a @ W[i] + b[i]
        a = np.maximum(0, z)
    return a @ W[-1] + b[-1]

def train_step(batch, W, b, Wt, bt, lr, gamma):
    # batch of (s,a,r,sn,done,mask_next)
    grads_W = [np.zeros_like(w) for w in W]
    grads_b = [np.zeros_like(bb) for bb in b]
    for (s,a,r,sn,d,mask_next) in batch:
        q = forward(s, W, b)
        with np.errstate(over='ignore'):
            q_next = forward(sn, Wt, bt) if not d else np.zeros_like(q)
        if mask_next is not None and not d:
            q_next = np.where(mask_next>0, q_next, -1e9)
        target = r if d else r + gamma * np.max(q_next)
        # backprop for selected action a
        # compute forward activations for backprop
        activations = [s]
        preacts = []
        a_in = s
        for i in range(len(W)-1):
            z = a_in @ W[i] + b[i]; preacts.append(z)
            a_in = np.maximum(0, z)
            activations.append(a_in)
        out = a_in @ W[-1] + b[-1]
        # grad on output layer
        dq = np.zeros_like(out)
        dq[a] = -1.0
        grad_out = (out[a] - target)  # scalar
        dq[a] = grad_out
        # grads for last layer
        grads_W[-1] += np.outer(activations[-1], dq)
        grads_b[-1] += dq
        # backprop through relu layers
        grad = dq @ W[-1].T
        for li in reversed(range(len(W)-1)):
            relu_mask = (preacts[li] > 0).astype(float)
            grad = grad * relu_mask
            grads_W[li] += np.outer(activations[li], grad)
            grads_b[li] += grad
            if li>0:
                grad = grad @ W[li].T
    # apply update
    for i in range(len(W)):
        W[i] -= lr * grads_W[i] / len(batch)
        b[i] -= lr * grads_b[i] / len(batch)

class DQNAgent:
    def __init__(self, obs_dim, n_actions, lr=5e-4, gamma=0.97, eps_start=0.3, eps_end=0.02, eps_decay=0.999):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.W, self.b = mlp([obs_dim, 256, 128, n_actions])
        self.Wt = [w.copy() for w in self.W]
        self.bt = [bb.copy() for bb in self.b]
        self.lr=lr; self.gamma=gamma
        self.eps=eps_start; self.eps_end=eps_end; self.eps_decay=eps_decay
        self.replay = Replay(75000)
        self.update_ctr = 0

    def act(self, s, legal_mask=None):
        if random.random() < self.eps:
            a = random.randrange(self.n_actions)
        else:
            q = forward(s, self.W, self.b)
            if legal_mask is not None:
                q = np.where(legal_mask>0, q, -1e9)
            a = int(np.argmax(q))
        return a

    def step(self, batch_size=128):
        if len(self.replay) < batch_size: return
        batch = self.replay.sample(batch_size)
        train_step(batch, self.W, self.b, self.Wt, self.bt, self.lr, self.gamma)
        self.update_ctr += 1
        if self.update_ctr % 100 == 0:
            self.Wt = [w.copy() for w in self.W]
            self.bt = [bb.copy() for bb in self.b]
        self.eps = max(self.eps_end, self.eps*self.eps_decay)

    def save(self, path):
        data = {"W":[w.tolist() for w in self.W],"b":[bb.tolist() for bb in self.b],"eps":self.eps}
        with open(path, "w") as f: json.dump(data,f)

# ============== Observation & Actions ==============
# Actions: 0=Skip, 1=Buy(if unowned), 2=Build(one house), 3=OfferTrade, 4=PayToLeaveJail
N_ACTIONS = 5

def build_obs(state: GameState, pid: int):
    p = state.players[pid]
    own = np.zeros(40); opp = np.zeros(40); houses = np.zeros(40); mort = np.zeros(40)
    for i in range(40):
        owner = state.owner.get(i,-1)
        if owner==pid:
            own[i]=1; houses[i]=p.houses.get(i,0)/4.0; mort[i]=1.0 if p.mortgaged.get(i,False) else 0.0
        elif owner!=-1:
            opp[i]=1
    pos = np.zeros(40); pos[p.pos]=1
    cash = np.array([p.cash/4000.0])
    jail = np.array([1.0 if p.jail_turns>0 else 0.0])
    sets = np.array([sum(1 for c,g in COLOR_GROUPS.items() if c not in ("rail","util") and has_full_set(pid,c,state))/10.0])
    return np.concatenate([own, opp, houses, mort, pos, cash, jail, sets])

def legal_mask(state: GameState, pid: int):
    p = state.players[pid]
    mask = np.ones(N_ACTIONS, dtype=int)
    # Buy legal only if standing on unowned purchasable and have cash
    can_buy = 0
    t = TILES[p.pos]
    if t.ttype in (PROP, RR, UTIL) and state.owner.get(p.pos,-1)==-1 and p.cash >= t.price:
        can_buy=1
    # Build legal if can build
    can_build_flag = 1 if can_build(pid, state) is not None else 0
    # Trade always allowed if both sides have something, otherwise off
    can_trade = 1 if (len(p.properties)>0 and any(len(o.properties)>0 for o in state.players if o.pid!=pid and not o.bankrupt)) else 0
    # Pay to leave jail allowed only if in jail and have >=50
    pay_jail = 1 if (p.jail_turns>0 and p.cash>=50) else 0
    mask = np.array([1, can_buy, can_build_flag, can_trade, pay_jail], dtype=int)
    return mask

# ============== Scripted Opponents ==============
def scripted_action(state: GameState, pid: int, style="aggressive"):
    mask = legal_mask(state, pid)
    p = state.players[pid]
    if style=="aggressive":
        if mask[1]: return 1
        if mask[2] and random.random()<0.7: return 2
        if mask[3] and random.random()<0.2: return 3
        if mask[4] and random.random()<0.5: return 4
        return 0
    if style=="conservative":
        if mask[2] and random.random()<0.4: return 2
        if mask[1] and p.cash>500: return 1
        if mask[3] and random.random()<0.1: return 3
        if mask[4] and random.random()<0.8: return 4
        return 0
    if style=="builder":
        if mask[2]: return 2
        if mask[1] and random.random()<0.6: return 1
        if mask[3] and random.random()<0.15: return 3
        if mask[4] and random.random()<0.6: return 4
        return 0
    return 0

# ============== Game Loop (One Episode) ==============
def step_player(state: GameState, pid: int, action: int):
    p = state.players[pid]
    pre_worth = [pp.net_worth() for pp in state.players]
    # Jail handling
    if p.jail_turns>0:
        if action==4 and p.cash>=50:
            p.cash -= 50; p.jail_turns=0
        else:
            # attempt roll doubles to get out; else decrement timer and skip rest
            dice, dbl, dice_pair = roll_two()
            if dbl:
                p.jail_turns=0
                prev = p.pos
                p.pos = (p.pos + dice) % BOARD_SIZE
                if p.pos < prev: p.cash += GO_BONUS
                _ = handle_tile(state, pid, dice)
            else:
                p.jail_turns -= 1
                if p.jail_turns==0:
                    if p.cash>=50: p.cash-=50
                auto_mortgage_if_needed(state, p)
                post_worth = [pp.net_worth() for pp in state.players]
                return post_worth, False
    else:
        # Normal move
        dice, dbl, dice_pair = roll_two()
        prev = p.pos
        p.pos = (p.pos + dice) % BOARD_SIZE
        if p.pos < prev: p.cash += GO_BONUS
        buy_signal = handle_tile(state, pid, dice)
        # Option actions:
        if action==1 and buy_signal=="UNOWNED":
            t = TILES[p.pos]
            if p.cash >= t.price:
                p.cash -= t.price
                p.properties.append(p.pos)
                state.owner[p.pos]=pid
            else:
                auction_property(state, p.pos, pid)
        elif action==2:
            bi = can_build(pid, state)
            if bi is not None:
                t = TILES[bi]
                p.cash -= t.house_cost
                p.houses[bi] = p.houses.get(bi,0)+1
        elif action==3:
            propose_trade(state, p)
        # auto mortgage if negative
        auto_mortgage_if_needed(state, p)
    post_worth = [pp.net_worth() for pp in state.players]
    done = (sum((not pp.bankrupt) for pp in state.players) <= 1) or (state.move_count>=MAX_MOVES)
    return post_worth, done

def play_episode(agent, opponents="scripted", seed=None, log_eval=False):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    state = GameState(n_players=4)
    done=False
    obs_prev = None; act_prev = None; pid_prev=None
    learner_pid = 0
    state.turn = 0
    total_rewards = np.zeros(4, dtype=float)
    while not done:
        p = state.players[state.turn]
        if p.bankrupt:
            state.next_turn(); continue
        obs = build_obs(state, state.turn)
        mask = legal_mask(state, state.turn)
        if state.turn == learner_pid:
            a = agent.act(obs, legal_mask=mask)
        else:
            if opponents=="scripted":
                a = scripted_action(state, state.turn, style=random.choice(["aggressive","builder","conservative"]))
            else:
                qs = forward(obs, agent.W, agent.b)
                qs = np.where(mask>0, qs, -1e9)
                a = int(np.argmax(qs))
                if random.random()<0.2: a = random.choice([i for i in range(5) if mask[i]])
        pre_worths = [pp.net_worth() for pp in state.players]
        post_worths, term = step_player(state, state.turn, a)
        r = post_worths[state.turn] - pre_worths[state.turn]
        total_rewards[state.turn]+=r
        state.move_count+=1
        if pid_prev is not None and pid_prev==learner_pid:
            agent.replay.push(obs_prev, act_prev, r_prev, obs, term, None)
        if state.turn==learner_pid:
            obs_prev = obs; act_prev = a; r_prev = r; pid_prev = state.turn
        else:
            pid_prev=None
        done = term
        state.next_turn()
    if pid_prev is not None and pid_prev==learner_pid:
        agent.replay.push(obs_prev, act_prev, r_prev, obs, True, None)
    alive = [p for p in state.players if not p.bankrupt]
    winner = alive[0].pid if len(alive)==1 else -1
    return {
        "winner": winner,
        "wealth": [p.net_worth() for p in state.players],
        "landings": state.landing_hist.tolist(),
        "total_rewards": total_rewards.tolist(),
        "moves": state.move_count
    }

def train(args):
    ensure_dir(args.save_dir)
    dummy_state = GameState(4)
    obs_dim = len(build_obs(dummy_state, 0))
    agent = DQNAgent(obs_dim, 5, lr=5e-4, gamma=0.97, eps_start=0.3, eps_end=0.02, eps_decay=0.999)
    csv_path = os.path.join(args.save_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode","winner","moves","agent_eps","avg_wealth","agent_wealth","eval_winrate","eval_avg_moves"])
    eval_stats = {"winrate":None,"avg_moves":None}
    for ep in range(1, args.episodes+1):
        stats = play_episode(agent, opponents=args.opponents)
        for _ in range(4): agent.step(128)
        if ep % args.eval_every == 0:
            wins = 0; moves = []
            for _ in range(50):
                st = play_episode(agent, opponents=args.opponents)
                wins += (st["winner"]==0)
                moves.append(st["moves"])
            eval_stats["winrate"] = wins/50
            eval_stats["avg_moves"] = float(np.mean(moves))
            agent.save(os.path.join(args.save_dir, f"weights_ep{ep}.json"))
        avg_wealth = float(np.mean(stats["wealth"]))
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, stats["winner"], stats["moves"], round(agent.eps,4), round(avg_wealth,1), stats["wealth"][0], eval_stats["winrate"], eval_stats["avg_moves"]])
        if ep % args.eval_every == 0:
            print(f"[{ep}] eps={agent.eps:.3f} eval_winrate={eval_stats['winrate']:.2f} avg_moves={eval_stats['avg_moves']:.1f}")
    agent.save(os.path.join(args.save_dir, "weights_final.json"))
    print("Done. Logs at:", csv_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--opponents", type=str, choices=["scripted","selfplay"], default="scripted")
    parser.add_argument("--save-dir", type=str, default="./mono_adv")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()

