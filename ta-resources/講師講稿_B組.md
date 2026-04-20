# Instructor Script — RL Lab, Group B (Colab / Jupyter)

> **使用說明**
> `【動作】` = 你的行為提示，**不念出來**。
> 引號內是**口說英文講詞**，可照念或自由調整語氣。
> **粗體**是需要念慢、讓學生記住的關鍵詞。

---

## ══ Day 1 ══

---

### 🖥️ Slide 1 — Title (1 min)

【開 slides_B組.html，全螢幕投影。第 1 張：標題頁】
【站到講台正中，等學生安靜，環視一圈再開口】

"Hello everyone. My name is **Colombo Chao** — you can call me Colombo. I'm a graduate researcher at Yuan Ze University, and today I'll be your instructor for this two-day lab."

"This is also part of my research. Your work and observations are data for my thesis. So please take each task seriously — what you do here genuinely matters."

---

### 🖥️ Slide 2 — Course Overview (2 min)

【按 → 切到第 2 張：兩天課程結構】

"Two days, five tasks. The structure is on the screen."

"For each task: I'll play a short demo video, then you do it yourself. When you're done, raise your hand — the TA will come check your work."

"You don't need to write any code. You'll change a few numbers, run the cells, and observe what happens. But I want you to actually *think* about why the result looks the way it does."

"Questions go to the TA, or raise your hand for me. No question is a stupid one."

---

### 🖥️ Slide 3 — Poll (2 min)

【按 → 切到第 3 張：互動提問頁】

"Before we dive in — quick question."

"Have you ever studied reinforcement learning before? Raise your hand if yes."

【停，數一下舉手的人，點評】

"Okay. How about — heard of it, but never actually tried it?"

【停，數一下】

"And completely new to the topic?"

【停】

"Perfect. That's exactly the range I expected. Whether you've seen this before or not — today's format is hands-on, so everyone will be starting from the same place operationally."

---

### 🖥️ Slide 4 — What is RL? (2 min)

【按 → 切到第 4 張：Agent / Environment 圖】

"Here's the core idea — one picture, one sentence."

"There's an **agent** — the learner. There's an **environment** — the world it operates in. The agent takes an **action**, the environment responds with a new **state** and a **reward**. The agent uses that feedback to decide what to do next."

"That loop — action, feedback, adjust — repeated thousands of times, is how reinforcement learning works."

"Every task today is a version of this loop. You'll see it in action."

---

### 🖥️ Slide 5 — Repo + Environment Setup (3 min)

【按 → 切到第 5 張：QR code 頁】

"Alright — scan this QR code, or type the URL. This GitHub page is your guide for both days."

【等學生掃碼或打 URL，助教協助確認】

"Once you're on the page — open the Colab or Binder link under Section 2. Make sure RL_Day1.ipynb loads and you can see the code cells."

【等 90 秒，確認全員就緒。有問題的讓助教處理。】
【關閉投影片，後續直接在 README 頁面上操作】

"Good. Keep that page open — you'll use it all day."

---

### 📺 V0 · What is RL — SAR Loop & Episode

【播放 V0：https://www.youtube.com/watch?v=g5SFtsTAv4I】

【影片結束後說】

"That's the core loop you'll see in every task today: the agent observes a **State**, takes an **Action**, and receives a **Reward**. One full run — from start to a terminal condition — is called an **episode**. Everything we do today and tomorrow builds on this loop."

---

### 🎯 T1 · MAB — Multi-Armed Bandit (25 min)

#### Intro (2 min)

"The first task is called **Multi-Armed Bandit** — MAB for short."

"Imagine you walk into a casino. There are many slot machines, but you don't know which one pays out the most. How do you decide whether to keep playing the same machine, or try a different one? That tension — between **exploiting** what you already know and **exploring** what you don't — is one of the most fundamental problems in reinforcement learning."

#### 【播放 V1 影片】

【播放 V1：https://www.youtube.com/watch?v=GrPdk2d9KVA】

【影片結束後說】

"Three parameters to know. **ε** — epsilon — controls how often the agent explores versus sticking with what it already knows. **α** — learning rate — controls how fast Q-values update each step. **γ** — discount factor — controls how much future reward matters compared to immediate reward. For this task, ε is your main variable. The other two will come up naturally as we go."

#### 【播放 B1 影片】

【全螢幕播放 B1 影片：https://youtu.be/GbpV4j7cR-Y】

【影片結束後繼續說】

"So the key parameter is **ε — epsilon**. ε = 0.9 means the agent explores 90% of the time. ε = 0.1 means it mostly sticks with what it thinks is best. Your task is to run both and compare the reward curves."

"**Task T1**: Run ε = 0.9 and ε = 0.1. Describe the difference in the reward curve. Which one learns better, and why?"

#### 【學生操作，走下講台巡視 18 min】

【卡在「跑不出圖」→ 提示：確認有按 Shift+Enter 執行格子】
【問「要設多少 ε」→ 說：就這兩個值，一個一個跑】
【完成的學生 → 追問：Which curve rises faster? What does that tell you about exploration?】

#### Wrap-up (3 min)

"Here's the takeaway: **exploration has a cost, but without exploration there's no learning**. High ε finds better options eventually, but wastes time trying bad ones. Low ε converges fast but might miss something better. Neither is always right — it depends on the problem."

"You'll see this trade-off in every task this week."

---

### 🎯 T2 · Maze 1D — Q-table & Bellman Update (25 min)

#### Intro (2 min)

"Next: a maze. The simplest possible one — a straight line. The agent moves left or right to find the goal."

"This time the agent doesn't just choose actions — it learns to **evaluate positions**. It builds a table of values: for every position, how good is it to be here? That's the **Q-table**."

"Every step produces three things: a **State**, an **Action**, and a **Reward** — S, A, R. You'll see them printed in the output. Your job is to recognize them."

#### 【播放 V2 影片】

【播放 V2：https://www.youtube.com/watch?v=lwEo9spItjs】

【影片結束後說】

"So the Q-table is the agent's memory — one value for every state-action pair. The **Bellman update** is the rule: each step, nudge that value a little closer to what actually happened — immediate reward plus discounted future value. Enough steps, and the whole table converges. Now watch the demo and look for this process happening live."

#### 【播放 B2 影片】

【播放 B2 影片：https://youtu.be/WRCE0S4DbZg】

【影片結束後】

"**Task T2**: Find the S, A, R in the output — point them out. Then explain in your own words what the **Bellman update** is doing: how did this step change the Q-value for that position?"

"No math required. Just explain the logic."

#### 【學生操作，巡視 18 min】

【看不懂 Q-value slice 圖 → 提示：X-axis is position along the maze. Y-axis is the expected reward from that position going right. Higher = closer to the goal.】
【問 Bellman 怎麼說 → 提示：The value of this state = immediate reward + discounted best future value. Say that in your own words.】

#### Wrap-up (3 min)

"**The Bellman equation** is the engine of Q-learning. Every step nudges the Q-value a little closer to the truth. Run enough steps, and the table converges. Everything else in RL — including deep neural networks — is built on this idea."

---

### 📺 V3 · Reading the Q-table Heatmap *(Day 2 preview)*

【播放 V3：https://www.youtube.com/watch?v=Sesod0K4wjc】

【影片結束後說】

"Tomorrow you'll build one of these yourself in the 2D maze. **Bright cells** mean the agent has learned that position leads to reward. **Arrows** show the preferred direction. Keep that picture in mind — that's what we're building toward on Day 2."

---

### ☕ Break (10 min)

"Alright, that's Day 1. Take ten minutes — water, bathroom, stretch."

【離開講台，讓學生真正放鬆】

---

## ══ Day 2 ══

---

### 🟢 Opening (2 min)

"Welcome back. Last week you covered two tasks — MAB and Maze 1D. Today we go 2D, then up to continuous state spaces."

"The progression is: a 2D maze first, to consolidate the Q-table and heatmap concept — then helicopter and fighter, where the state space becomes infinite. Each step builds on the last."

---

### 🎯 T3 · Maze 2D — Q-table Heatmap (30 min)

#### Intro (2 min)

"First task today: a 2D maze. The agent can move up, down, left, right."

"Why is this harder than Maze 1D? **The state space is much larger.** A 1D maze had maybe 20 positions. A 2D maze has hundreds of cells, and each one needs to be learned separately."

"After training, you'll see a **heatmap**: brighter color means the agent thinks that cell is more valuable. Arrows show the preferred action at each cell."

#### 【播放影片】

【播放 B3 影片：https://youtu.be/wFAx0JIwvZI】

【影片結束後】

"**Task T3**: On the heatmap, trace the path from Start to Goal. Follow the bright cells and arrows. Then explain — why are certain cells brighter than others?"

#### 【學生操作，巡視 23 min】

【訓練時間較長，提醒學生等輸出最底部的熱圖出現再截圖】
【熱圖沒出現 → 前一格還沒跑完，等 kernel 不再顯示 [*]】
【進階 → 讓學生改 MAZE2D_LEVEL，觀察路徑變化】

#### Wrap-up (2 min)

"The heatmap *is* the Q-table — just visualized. It lets you see inside the agent's head: where it thinks is worth going, and where it doesn't. That kind of interpretability is rare in machine learning. Enjoy it while it lasts."

"Now we step up: **continuous state spaces**."

"What does that mean? This maze had discrete positions — a finite number of cells. The next environments use **real-valued states**: position, velocity, angle — infinitely many possible values. How does the agent learn?"

"The answer is **discretization**: slice the continuous space into bins, then apply the Q-table you already know. But the finer the bins, the more the agent has to learn. You'll feel that trade-off directly."

---

### 🎯 T4 · Heli — Reading Training Curves (30 min)

#### Intro (2 min)

"Task 4 is a helicopter game — fly horizontally, dodge obstacles. **Five continuous state dimensions**: position, velocity, and so on."

"The main focus today isn't just running the agent — it's **reading the training curve**. That graph of reward over episodes: what is it telling you? Is the agent improving? Plateauing? Oscillating?"

#### 【播放 V4 影片】

【播放 V4：https://www.youtube.com/watch?v=6crIH-kT-bA】

【影片結束後說】

"Three patterns to recognize: **rising** — the agent is learning. **Flat** — it's plateaued, might need more episodes or a lower learning rate. **Noisy or oscillating** — something's unstable, maybe ε or α is too high. Your job this task: describe which pattern you see and give your best explanation for why."

#### 【播放 B4 影片】

【播放 B4 影片：https://youtu.be/l7TCZ9fFzNY】

【影片結束後】

"**Task T4**: Run at least 50 episodes. Then describe the reward curve trend — rising, flat, or noisy? Give your best explanation for why it looks that way."

"Rising = the agent is learning. Flat or noisy = it might need more time, or the parameters might need tuning."

#### 【學生操作，巡視 23 min】

【訓練時間長，提醒學生耐心等待】
【曲線很亂 → 正常。Look at the smoothed moving average, not individual points.】
【鼓勵學生嘗試改 BINS 值（格子數），觀察學習速度差異】

#### Wrap-up (3 min)

"**The training curve is your diagnostic tool.** A flat curve doesn't mean failure — it might just need more episodes. A wild curve might mean the learning rate is too high. Reading these graphs is a skill, and you practiced it today."

---

### 🎯 T5 · Fighter — Optional Challenge (open-ended)

#### Intro (2 min)

"The last one: a fighter jet game. Shoot rocks, dodge them, survive. This is the hardest environment — **5-dimensional continuous state**, 4 actions, 5 difficulty modes."

"This task is **optional**. If you haven't finished T4, keep working on that. If T4 is done, come try this."

#### 【播放 V5 影片】

【播放 V5：https://www.youtube.com/watch?v=Z67UnKtgBH4】

【影片結束後說】

"The key idea: a continuous state space has infinitely many values — too many to put in a Q-table directly. So we **discretize**: chop the continuous range into bins, treat each bin as a discrete state. Fewer bins = faster to learn, coarser resolution. More bins = precise, but the agent needs far more episodes. Fighter is where you feel that trade-off directly."

#### 【播放 B5 影片】

【播放 B5 影片：https://youtu.be/BzPVMk7DuqE】

【影片結束後】

"No fixed requirement — **explore freely**. Try different values of FIGHTER_MODE. See how the agent's behavior changes across difficulty levels. If you notice something interesting, write it down."

#### 【學生自由探索 15 min，走動個別聊觀察】

---

### 🎤 Closing Discussion (10 min)

【不需要投影，站台前即可】

"Before we wrap up, a few questions. No right answers — just think out loud."

"**How was this different from what you imagined AI learning would look like?**"

【等 2-3 個學生回答，追問：Why did you expect that?】

"**Which task do you think was hardest for the agent — and why?**"

【引導：state space size / sparse rewards / exploration difficulty】

"Here's the bigger picture: everything you did this week — Q-table, Bellman updates, exploration vs exploitation — these are the same core ideas behind the systems you hear about in the news. Scale them up with deep neural networks, and you get the RLHF training that fine-tunes ChatGPT. You've touched the foundation."

"Thank you. The TA will collect the forms. Feel free to ask questions before you leave."

【下課】

---

## 📎 Video Links

| Task | Topic | Link |
|------|-------|------|
| B1 | MAB — Exploration vs Exploitation | https://youtu.be/GbpV4j7cR-Y |
| B2 | Maze 1D — SAR & Q-table | https://youtu.be/WRCE0S4DbZg |
| B3 | Maze 2D — Q-table Heatmap | https://youtu.be/wFAx0JIwvZI |
| B4 | Heli — Training Curves | https://youtu.be/l7TCZ9fFzNY |
| B5 | Fighter — Optional | https://youtu.be/BzPVMk7DuqE |

---

## 📎 Troubleshooting

| Problem | Fix |
|---------|-----|
| Jupyter won't start | 請助教用 `jupyter notebook` 重開；確認在正確資料夾 |
| Cell won't run | 確認按 Shift+Enter；確認前一格 [*] 已跑完 |
| Missing packages | Run `!pip install gymnasium matplotlib numpy` in the first cell |
| Training is slow | Reduce EPISODES (try 50 first) |
| Plot doesn't appear | 等 kernel 完成（cell 左側從 [*] 變成 [數字]） |
| "What's the real-world use?" | "This is the same algorithm, scaled up with neural nets, that trains ChatGPT's RLHF layer." |
