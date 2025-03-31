# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import time
from collections import defaultdict
from ortools.sat.python import cp_model
from itertools import combinations
import numpy as np

st.set_page_config(layout="wide")

# --- 고정 설정 및 데이터 ---
NUM_DAYS = 3
GAMES_PER_DAY = [4, 3, 3]
NUM_GAMES = sum(GAMES_PER_DAY)
EXACT_TIER1_PER_TEAM_FIXED = 1
MIN_ENEMY_SAME_POS_FIXED = 1

positions = ['T', 'J', 'M', 'A', 'S']
players_per_team = len(positions)
num_teams_per_game = 2

player_data = {
    'T': {1: ('인섹', 'T1'), 2: ('얍얍', 'T2'), 3: ('뽀융쨩', 'T3'), 4: ('룩삼', 'T4'), 5: ('강소연', 'T5')},
    'J': {1: ('소우릎', 'J1'), 2: ('꼴랑이', 'J2'), 3: ('네클릿', 'J3'), 4: ('휘용', 'J4'), 5: ('고수달', 'J5')},
    'M': {1: ('갱맘', 'M1'), 2: ('실프', 'M2'), 3: ('헤징', 'M3'), 4: ('젤리', 'M4'), 5: ('명훈', 'M5')},
    'A': {1: ('크캣', 'A1'), 2: ('따효니', 'A2'), 3: ('플러리', 'A3'), 4: ('모카형', 'A4'), 5: ('러너', 'A5')},
    'S': {1: ('라콩', 'S1'), 2: ('눈꽃', 'S2'), 3: ('던', 'S3'), 4: ('루루카', 'S4'), 5: ('감규리', 'S5')},
}

@st.cache_data
def process_player_data(p_data):
    players = []; player_alias = {}; player_pos = {}; player_rank = {}
    pos_players = defaultdict(list); tier1_players = []
    player_to_id = {}; id_to_player = {}; pid_counter = 0; all_player_names = []
    display_player_options = []
    display_to_name_map = {}
    name_to_display_map = {}

    for pos, ranks in p_data.items():
        for rank, (alias, name) in ranks.items():
            players.append(name); player_alias[name] = alias; player_pos[name] = pos
            player_rank[name] = rank; pos_players[pos].append(name); all_player_names.append(name)
            display_text = f"{alias}({name})"
            display_player_options.append(display_text)
            display_to_name_map[display_text] = name
            name_to_display_map[name] = display_text
            if rank == 1: tier1_players.append(name)
            player_to_id[name] = pid_counter; id_to_player[pid_counter] = name; pid_counter += 1

    num_players = len(players); player_ids = list(range(num_players))
    non_tier1_players = [p for p in players if p not in tier1_players]
    return (players, player_alias, player_pos, player_rank, pos_players,
            tier1_players, player_to_id, id_to_player, num_players,
            player_ids, sorted(all_player_names), non_tier1_players,
            sorted(display_player_options), display_to_name_map, name_to_display_map)

(players, player_alias, player_pos, player_rank, pos_players,
 tier1_players, player_to_id, id_to_player, num_players,
 player_ids, all_player_names, non_tier1_players,
 display_player_options, display_to_name_map, name_to_display_map) = process_player_data(player_data)


def get_player_info(p_id):
    name = id_to_player.get(p_id, "Unknown")
    if name == "Unknown": return "Unknown", "Unk", "N/A", -1
    return name, player_alias.get(name,"?"), player_pos.get(name,"?"), player_rank.get(name,-1)

# --- OR-Tools 스케줄링 로직 ---
def solve_schedule(positions, player_data,
                   num_teams_per_game, players_per_team,
                   banned_players_by_day,
                   time_limit_seconds):
    num_games = NUM_GAMES; num_days = NUM_DAYS
    exact_tier1_per_team = EXACT_TIER1_PER_TEAM_FIXED
    min_enemy_same_pos_diff_rank = MIN_ENEMY_SAME_POS_FIXED

    (players_f, player_alias_f, player_pos_f, player_rank_f, pos_players_f, tier1_players_f,
     player_to_id_f, id_to_player_f, num_players_f, player_ids_f, _, non_tier1_players_f,
     _, display_to_name_map_f, _) = process_player_data(player_data)
    tier1_player_ids_f = {player_to_id_f[p] for p in tier1_players_f}
    non_tier1_player_ids_f = {player_to_id_f[p] for p in non_tier1_players_f}

    status_messages = []; status_area = st.empty()
    def update_status(msg): status_messages.append(msg); status_area.text("\n".join(status_messages))

    update_status(f"\n=== {num_games} 게임 스케줄 생성 시작 (0회 매치업 최소화, 시간 제한: {time_limit_seconds}초) ===")
    start_time = time.time(); model = cp_model.CpModel()

    # --- 선수 ID 집합 생성 ---
    update_status("[계산] 선수 ID 집합 생성 중...");
    banned_player_ids_by_day = defaultdict(set)
    for day, banned_names_display in banned_players_by_day.items():
        day_idx = day - 1
        if 0 <= day_idx < num_days:
            banned_names_actual = {display_to_name_map_f.get(disp_name) for disp_name in banned_names_display}
            banned_player_ids_by_day[day_idx] = {player_to_id_f[p] for p in banned_names_actual if p in player_to_id_f}
    pos_player_ids = defaultdict(list); [pos_player_ids[pos].append(player_to_id_f[p]) for pos, p_list in pos_players_f.items() for p in p_list]
    update_status("[계산] 선수 ID 집합 생성 완료.")

    # --- 비1티어 목표 경기 수 계산 ---
    total_non_tier1_slots = num_games * num_teams_per_game * (players_per_team - 1)
    num_non_tier1_players = len(non_tier1_player_ids_f)
    target_play_count_non_tier1 = -1
    if num_non_tier1_players > 0 and total_non_tier1_slots % num_non_tier1_players == 0:
         target_play_count_non_tier1 = total_non_tier1_slots // num_non_tier1_players
         update_status(f"[계산] 비1티어 목표 경기 수: {target_play_count_non_tier1}")
    elif num_non_tier1_players > 0:
         msg = f"[경고] 비1티어 선수({num_non_tier1_players}명) 수로 총 슬롯({total_non_tier1_slots}개)을 나눌 수 없어 동일 경기 수 제약 제외됨."
         update_status(msg)
    else:
         msg = "[경고] 비1티어 선수가 없어 동일 경기 수 제약 제외됨."
         update_status(msg)

    # --- 모델 변수 생성 ---
    update_status("[모델 생성] 변수 생성 중..."); pos_indices = list(range(len(positions))); pos_map = {pos: i for i, pos in enumerate(positions)}
    assignment = {}; valid_model = True
    for g in range(num_games):
        for t in range(num_teams_per_game):
            for p_idx in pos_indices:
                pos = positions[p_idx]; allowed_player_ids_for_pos = pos_player_ids.get(pos); var_key = (g, t, p_idx)
                if not allowed_player_ids_for_pos: msg = f"[오류] '{pos}' 선수 없음!"; update_status(msg); valid_model = False; break
                domain = cp_model.Domain.FromValues(allowed_player_ids_for_pos)
                assignment[var_key] = model.NewIntVarFromDomain(domain, f'assign_g{g}_t{t}_p{pos}')
            if not valid_model: break
        if not valid_model: break
    if not valid_model: return None, None, status_messages, cp_model.MODEL_INVALID
    update_status(f"[모델 생성] assignment 변수 {len(assignment)}개 생성.")

    player_in_game = {}; [player_in_game.setdefault((p_id, g), model.NewBoolVar(f'p{p_id}_in_g{g}')) for p_id in player_ids_f for g in range(num_games)]
    update_status(f"[모델 생성] player_in_game 변수 {len(player_in_game)}개 생성.")
    game_day = [model.NewIntVar(0, num_days - 1, f'game_day_{g}') for g in range(num_games)]
    update_status(f"[모델 생성] game_day 변수 {len(game_day)}개 생성.")

    # --- 변수 연결 제약 ---
    update_status("[모델 생성] 변수 연결 제약 추가 중...");
    for p_id in player_ids_f:
        name = id_to_player_f.get(p_id)
        p_pos = player_pos_f.get(name) if name else None
        p_pos_idx = pos_map.get(p_pos) if p_pos else None
        if p_pos_idx is None: continue
        for g in range(num_games):
            presence_indicators = []
            for t in range(num_teams_per_game):
                 key = (g, t, p_pos_idx)
                 if key in assignment:
                     indicator = model.NewBoolVar(f'ind_p{p_id}_g{g}_t{t}_pos{p_pos_idx}')
                     model.Add(assignment[key] == p_id).OnlyEnforceIf(indicator); model.Add(assignment[key] != p_id).OnlyEnforceIf(indicator.Not()); presence_indicators.append(indicator)
            if (p_id, g) in player_in_game and presence_indicators:
                model.Add(sum(presence_indicators) >= 1).OnlyEnforceIf(player_in_game[p_id, g]);
                model.Add(sum(presence_indicators) == 0).OnlyEnforceIf(player_in_game[p_id, g].Not())
            elif (p_id, g) in player_in_game:
                 model.Add(player_in_game[p_id, g] == 0)
    update_status("[모델 생성] 변수 연결 제약 추가 완료.")

    # --- 주요 제약 조건 추가 ---
    update_status("\n--- 주요 제약 조건 추가 시작 ---"); constraint_count = 0
    # (C2) 게임 내 중복 금지
    update_status("[제약 추가 중] (C2) 게임 내 중복 금지..."); count_c2 = 0
    for g in range(num_games):
        players_in_game_g = [assignment[g, t, p_idx] for t in range(num_teams_per_game) for p_idx in pos_indices if (g,t,p_idx) in assignment]
        if len(players_in_game_g) > 1: model.AddAllDifferent(players_in_game_g); count_c2 += 1
    constraint_count += count_c2; update_status(f"[제약 추가 완료] (C2) {count_c2}개 추가.")

    # (C3) 팀당 1티어 선수 수
    update_status("[제약 추가 중] (C3) 팀당 1티어 선수 수..."); count_c3 = 0
    for g in range(num_games):
        for t in range(num_teams_per_game):
            is_tier1_flags = []
            for p_idx in pos_indices:
                 key = (g, t, p_idx); pos = positions[p_idx]
                 if key not in assignment: continue
                 is_t1 = model.NewBoolVar(f'isT1_g{g}_t{t}_p{p_idx}'); pos_tier1_ids = [p_id for p_id in pos_player_ids.get(pos, []) if p_id in tier1_player_ids_f]
                 if not pos_tier1_ids: model.Add(is_t1 == 0)
                 else:
                     t1_matches = []
                     for t1_id in pos_tier1_ids:
                         match = model.NewBoolVar(f'm_g{g}_t{t}_p{p_idx}_{t1_id}')
                         model.Add(assignment[key] == t1_id).OnlyEnforceIf(match); model.Add(assignment[key] != t1_id).OnlyEnforceIf(match.Not()); t1_matches.append(match)
                     model.AddBoolOr(t1_matches).OnlyEnforceIf(is_t1)
                     model.Add(sum(t1_matches) == 0).OnlyEnforceIf(is_t1.Not())
                 is_tier1_flags.append(is_t1);
            if is_tier1_flags: model.Add(sum(is_tier1_flags) == exact_tier1_per_team); count_c3 += 1
    constraint_count += count_c3; update_status(f"[제약 추가 완료] (C3) {count_c3}개 팀 제약 추가.")

    # --- 아군/적군 보조 변수 생성 ---
    update_status("[모델 생성] 아군/적군 조건용 변수 생성 중..."); are_enemies = {}; are_allies = {}
    for g in range(num_games):
        for i, p1_id in enumerate(player_ids_f):
            for j in range(i + 1, num_players_f):
                p2_id = player_ids_f[j];
                id1, id2 = min(p1_id, p2_id), max(p1_id, p2_id)

                p1_plays = player_in_game.get((p1_id, g), 0); p2_plays = player_in_game.get((p2_id, g), 0)
                both_play = model.NewBoolVar(f'both_{id1}_{id2}_g{g}');
                enemies_var = model.NewBoolVar(f'enemy_{id1}_{id2}_g{g}');
                allies_var = model.NewBoolVar(f'ally_{id1}_{id2}_g{g}')
                are_enemies[id1, id2, g] = enemies_var
                are_allies[id1, id2, g] = allies_var

                p1_lit = p1_plays if isinstance(p1_plays, cp_model.IntVar) else model.NewConstant(p1_plays);
                p2_lit = p2_plays if isinstance(p2_plays, cp_model.IntVar) else model.NewConstant(p2_plays)

                if isinstance(p1_lit, cp_model.IntVar) and isinstance(p2_lit, cp_model.IntVar):
                    model.AddBoolAnd([p1_lit, p2_lit]).OnlyEnforceIf(both_play);
                    model.AddBoolOr([p1_lit.Not(), p2_lit.Not()]).OnlyEnforceIf(both_play.Not())
                elif p1_lit == 0 or p2_lit == 0: model.Add(both_play == 0)
                else: model.Add(both_play == 1)

                model.Add(enemies_var == 0).OnlyEnforceIf(both_play.Not())
                model.Add(allies_var == 0).OnlyEnforceIf(both_play.Not())

                p1_name = id_to_player_f.get(p1_id); p2_name = id_to_player_f.get(p2_id);
                if not p1_name or not p2_name: continue
                p1_pos = player_pos_f.get(p1_name); p2_pos = player_pos_f.get(p2_name);
                if not p1_pos or not p2_pos: continue
                p1_pos_idx = pos_map.get(p1_pos); p2_pos_idx = pos_map.get(p2_pos);
                if p1_pos_idx is None or p2_pos_idx is None: continue

                # 아군 조건
                if p1_pos != p2_pos:
                    same_team_indicators = [];
                    for t in range(num_teams_per_game):
                        key1 = (g, t, p1_pos_idx); key2 = (g, t, p2_pos_idx);
                        if key1 in assignment and key2 in assignment:
                            p1_on_t = model.NewBoolVar(f'p1_t{t}_{g}_{id1}'); model.Add(assignment[key1] == p1_id).OnlyEnforceIf(p1_on_t); model.Add(assignment[key1] != p1_id).OnlyEnforceIf(p1_on_t.Not())
                            p2_on_t = model.NewBoolVar(f'p2_t{t}_{g}_{id2}'); model.Add(assignment[key2] == p2_id).OnlyEnforceIf(p2_on_t); model.Add(assignment[key2] != p2_id).OnlyEnforceIf(p2_on_t.Not())
                            same_t = model.NewBoolVar(f'same_t{t}_{id1}_{id2}_g{g}'); model.AddBoolAnd([p1_on_t, p2_on_t]).OnlyEnforceIf(same_t); model.AddBoolOr([p1_on_t.Not(), p2_on_t.Not()]).OnlyEnforceIf(same_t.Not())
                            same_team_indicators.append(same_t)
                    if same_team_indicators:
                        model.AddBoolOr(same_team_indicators).OnlyEnforceIf(allies_var)
                        model.Add(sum(same_team_indicators) == 0).OnlyEnforceIf(allies_var.Not())
                    else: model.Add(allies_var == 0)
                else: model.Add(allies_var == 0)

                # 적군 조건
                if num_teams_per_game == 2:
                    key1_t0=(g,0,p1_pos_idx); key1_t1=(g,1,p1_pos_idx);
                    key2_t0=(g,0,p2_pos_idx); key2_t1=(g,1,p2_pos_idx);
                    if key1_t0 in assignment and key1_t1 in assignment and key2_t0 in assignment and key2_t1 in assignment:
                        p1_t0 = model.NewBoolVar(f'p1_t0_{g}_{id1}'); model.Add(assignment[key1_t0]==p1_id).OnlyEnforceIf(p1_t0); model.Add(assignment[key1_t0]!=p1_id).OnlyEnforceIf(p1_t0.Not())
                        p1_t1 = model.NewBoolVar(f'p1_t1_{g}_{id1}'); model.Add(assignment[key1_t1]==p1_id).OnlyEnforceIf(p1_t1); model.Add(assignment[key1_t1]!=p1_id).OnlyEnforceIf(p1_t1.Not())
                        p2_t0 = model.NewBoolVar(f'p2_t0_{g}_{id2}'); model.Add(assignment[key2_t0]==p2_id).OnlyEnforceIf(p2_t0); model.Add(assignment[key2_t0]!=p2_id).OnlyEnforceIf(p2_t0.Not())
                        p2_t1 = model.NewBoolVar(f'p2_t1_{g}_{id2}'); model.Add(assignment[key2_t1]==p2_id).OnlyEnforceIf(p2_t1); model.Add(assignment[key2_t1]!=p2_id).OnlyEnforceIf(p2_t1.Not())
                        diff_c1 = model.NewBoolVar(f'diff1_{id1}_{id2}_g{g}'); model.AddBoolAnd([p1_t0, p2_t1]).OnlyEnforceIf(diff_c1); model.AddBoolOr([p1_t0.Not(), p2_t1.Not()]).OnlyEnforceIf(diff_c1.Not())
                        diff_c2 = model.NewBoolVar(f'diff2_{id1}_{id2}_g{g}'); model.AddBoolAnd([p1_t1, p2_t0]).OnlyEnforceIf(diff_c2); model.AddBoolOr([p1_t1.Not(), p2_t0.Not()]).OnlyEnforceIf(diff_c2.Not())
                        model.AddBoolOr([diff_c1, diff_c2]).OnlyEnforceIf(enemies_var);
                        model.Add(sum([diff_c1, diff_c2]) == 0).OnlyEnforceIf(enemies_var.Not())
                    else: model.Add(enemies_var == 0)
                else: model.Add(enemies_var == 0)

                model.AddImplication(enemies_var, both_play)
                model.AddImplication(allies_var, both_play)
                model.Add(allies_var + enemies_var <= 1)
    update_status("[모델 생성] 아군/적군 조건용 변수 생성 완료.")

    # (NEW) 1티어 맞대결 1번
    update_status("[제약 추가 중] (NEW) 1티어 맞대결 정확히 1번씩 발생..."); count_t1_match = 0
    tier1_pairs = list(combinations(tier1_player_ids_f, 2))
    if len(tier1_pairs) != 10: update_status(f"[경고] 1티어 조합 수 10 아님: {len(tier1_pairs)}");
    for p1_id, p2_id in tier1_pairs:
        id1, id2 = min(p1_id, p2_id), max(p1_id, p2_id)
        enemy_vars = [are_enemies.get((id1, id2, g), 0) for g in range(num_games)]
        valid_enemy_vars = [v for v in enemy_vars if isinstance(v, cp_model.IntVar)]
        if valid_enemy_vars: model.Add(sum(valid_enemy_vars) == 1); count_t1_match += 1
    constraint_count += count_t1_match; update_status(f"[제약 추가 완료] (NEW) 1티어 맞대결 제약 {count_t1_match}개 추가.")

    # (C5) 같은 포지션 간 최소 적군 조건
    update_status("[제약 추가 중] (C5) 같은 포지션 간 최소 적군 조건 (1회 고정)..."); count_c5 = 0
    for pos in positions:
        pos_p_ids = sorted(pos_player_ids.get(pos, []))
        for i, p1_id in enumerate(pos_p_ids):
            for j in range(i + 1, len(pos_p_ids)):
                p2_id = pos_p_ids[j];
                id1, id2 = min(p1_id, p2_id), max(p1_id, p2_id)
                enemy_vars = [are_enemies.get((id1, id2, g), 0) for g in range(num_games)];
                valid_enemy_vars = [v for v in enemy_vars if isinstance(v, cp_model.IntVar)]
                if valid_enemy_vars: model.Add(sum(valid_enemy_vars) >= min_enemy_same_pos_diff_rank); count_c5 += 1
    constraint_count += count_c5; update_status(f"[제약 추가 완료] (C5) {count_c5}개 추가.")

    # (C8) 비1티어 선수 동일 경기 수
    if target_play_count_non_tier1 != -1:
        update_status(f"[제약 추가 중] (C8) 비1티어 선수 동일 경기 수 ({target_play_count_non_tier1}회)..."); count_c8 = 0
        for p_id in non_tier1_player_ids_f:
            play_count_vars = [player_in_game.get((p_id, g), 0) for g in range(num_games)]
            valid_play_vars = [v for v in play_count_vars if isinstance(v, cp_model.IntVar)]
            if valid_play_vars: model.Add(sum(valid_play_vars) == target_play_count_non_tier1); count_c8 += 1
        constraint_count += count_c8; update_status(f"[제약 추가 완료] (C8) {count_c8}개 추가.")
    else:
         update_status(f"[제약 제외됨] (C8) 비1티어 선수 동일 경기 수 제약 조건 제외됨.")

    # (NEW) 날짜별 게임 수 균등 분배
    update_status("[제약 추가 중] (NEW) 날짜별 게임 수 균등 분배 (3~4 게임)..."); count_day_bal = 0
    game_on_day_vars = [[model.NewBoolVar(f'g{g}_on_d{d}') for g in range(num_games)] for d in range(num_days)]
    for g in range(num_games):
        for d in range(num_days):
            model.Add(game_day[g] == d).OnlyEnforceIf(game_on_day_vars[d][g]); model.Add(game_day[g] != d).OnlyEnforceIf(game_on_day_vars[d][g].Not()); count_day_bal += 2
        model.Add(sum(game_on_day_vars[d][g] for d in range(num_days)) == 1); count_day_bal += 1
    for d in range(num_days):
        games_this_day = [game_on_day_vars[d][g] for g in range(num_games)]
        model.Add(sum(games_this_day) >= 3); model.Add(sum(games_this_day) <= 4); count_day_bal += 2
    constraint_count += count_day_bal; update_status(f"[제약 추가 완료] (NEW) 날짜 균등 분배 제약 약 {count_day_bal}개 추가.")

    # (C7) 특정 날짜 출전 금지
    update_status("[제약 추가 중] (C7) 특정 날짜 출전 금지..."); count_c7 = 0
    for day_idx, banned_ids in banned_player_ids_by_day.items():
        if banned_ids:
            for p_id in banned_ids:
                for g in range(num_games):
                    if (p_id, g) in player_in_game:
                        player_plays_var = player_in_game[p_id, g]
                        if isinstance(player_plays_var, cp_model.IntVar):
                             model.AddImplication(game_on_day_vars[day_idx][g], player_plays_var.Not()); count_c7 += 1
    if count_c7 > 0: constraint_count += count_c7; update_status(f"[제약 추가 완료] (C7) {count_c7}개 추가.")
    else: update_status("[제약 추가 완료] (C7) 해당 제약 없음.")

    # <<< --- NEW: 동일 날짜 연속 경기 출전 금지 제약 조건 추가 --- >>>
    update_status("[제약 추가 중] (NEW) 동일 날짜 연속 경기 출전 금지..."); count_consecutive = 0
    # 인접한 두 게임이 같은 날짜인지 나타내는 변수 미리 생성
    same_day_vars = {}
    for g in range(num_games - 1):
        var = model.NewBoolVar(f'same_day_{g}_{g+1}')
        # game_day 변수를 사용하여 g와 g+1이 같은 날인지 확인
        model.Add(game_day[g] == game_day[g+1]).OnlyEnforceIf(var)
        model.Add(game_day[g] != game_day[g+1]).OnlyEnforceIf(var.Not())
        same_day_vars[g] = var
        count_consecutive += 2 # 변수 정의 제약 2개

    # 모든 선수와 인접 게임 쌍에 대해 제약 적용
    for p_id in player_ids_f:
        for g in range(num_games - 1):
            p_plays_g = player_in_game.get((p_id, g))
            p_plays_gplus1 = player_in_game.get((p_id, g + 1))
            same_day_g_gplus1 = same_day_vars[g]

            # player_in_game 변수가 모델에 실제로 존재하는지 확인 (이론상 항상 존재)
            if isinstance(p_plays_g, cp_model.IntVar) and isinstance(p_plays_gplus1, cp_model.IntVar):
                # 제약: p_plays_g + p_plays_gplus1 + same_day_g_gplus1 <= 2
                # 즉, 세 변수가 동시에 1이 될 수 없음 (같은 날 연속 출전 금지)
                model.Add(p_plays_g + p_plays_gplus1 + same_day_g_gplus1 <= 2)
                count_consecutive += 1
            # else: player_in_game 변수가 없거나 상수로 고정된 경우 (예: 밴) - 제약 추가 불필요

    constraint_count += count_consecutive
    update_status(f"[제약 추가 완료] (NEW) 동일 날짜 연속 경기 금지 제약 약 {count_consecutive}개 추가.")
    # <<< --- 연속 경기 금지 제약 조건 추가 완료 --- >>>


    # --- 최적화 목표 설정 ---
    update_status("[최적화 목표 설정] 0회 매치업 최소화...")
    never_enemies_vars = []
    never_allies_vars = []
    zero_matchups_count = 0
    for i, p1_id in enumerate(player_ids_f):
        for j in range(i + 1, num_players_f):
            p2_id = player_ids_f[j]
            id1, id2 = min(p1_id, p2_id), max(p1_id, p2_id)

            # 적군 0회 변수
            total_enemies_vars = [are_enemies.get((id1, id2, g), 0) for g in range(num_games)]
            valid_enemy_vars = [v for v in total_enemies_vars if isinstance(v, cp_model.IntVar)]
            if valid_enemy_vars:
                 total_enemies = model.NewIntVar(0, num_games, f'total_enemies_{id1}_{id2}')
                 model.Add(total_enemies == sum(valid_enemy_vars))
                 never_enemy = model.NewBoolVar(f'never_enemy_{id1}_{id2}')
                 model.Add(total_enemies == 0).OnlyEnforceIf(never_enemy)
                 model.Add(total_enemies >= 1).OnlyEnforceIf(never_enemy.Not())
                 never_enemies_vars.append(never_enemy)
                 zero_matchups_count += 1

            # 아군 0회 변수
            p1_name = id_to_player_f.get(p1_id); p2_name = id_to_player_f.get(p2_id)
            if p1_name and p2_name:
                p1_pos = player_pos_f.get(p1_name); p2_pos = player_pos_f.get(p2_name)
                is_p1_t1 = p1_id in tier1_player_ids_f; is_p2_t1 = p2_id in tier1_player_ids_f

                if p1_pos and p2_pos and p1_pos != p2_pos and not (is_p1_t1 and is_p2_t1):
                    total_allies_vars = [are_allies.get((id1, id2, g), 0) for g in range(num_games)]
                    valid_ally_vars = [v for v in total_allies_vars if isinstance(v, cp_model.IntVar)]
                    if valid_ally_vars:
                         total_allies = model.NewIntVar(0, num_games, f'total_allies_{id1}_{id2}')
                         model.Add(total_allies == sum(valid_ally_vars))
                         never_ally = model.NewBoolVar(f'never_ally_{id1}_{id2}')
                         model.Add(total_allies == 0).OnlyEnforceIf(never_ally)
                         model.Add(total_allies >= 1).OnlyEnforceIf(never_ally.Not())
                         never_allies_vars.append(never_ally)
                         zero_matchups_count += 1

    model.Minimize(sum(never_enemies_vars) + sum(never_allies_vars))
    update_status(f"[최적화 목표 설정 완료] 총 {zero_matchups_count}개의 0회 매치업 변수 고려.")

    # --- 솔버 실행 ---
    update_status(f"\n--- 총 약 {constraint_count}개의 주요 제약 추가 완료 ---")
    update_status("\n[솔버 실행] CP-SAT 솔버 해 찾기 시작..."); solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8; update_status(f"[솔버 설정] 병렬 워커 수: {solver.parameters.num_search_workers}")
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    update_status(f"[솔버 설정] 최대 실행 시간: {time_limit_seconds}초")

    status = cp_model.UNKNOWN
    try:
        spinner_message = f'CP-SAT 솔버가 해를 찾고 있습니다... (최대 {time_limit_seconds}초)'
        with st.spinner(spinner_message):
            status = solver.Solve(model)
    except Exception as e:
        update_status(f"[솔버 오류] {e}");
        return None, None, status_messages, cp_model.UNKNOWN

    end_time = time.time();
    update_status(f"\n[솔버 실행 완료] 소요 시간: {solver.WallTime():.2f}초 (요청 시간 제한: {time_limit_seconds}초)")
    update_status(f"[솔버 상태] 결과: {solver.StatusName(status)}")
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        objective_value = solver.ObjectiveValue()
        update_status(f"[솔버 결과] 목표 값 (0회 매치업 수): {objective_value}")
    elif status == cp_model.INFEASIBLE:
         update_status("[솔버 결과] 모델이 비현실적입니다 (제약 조건을 만족하는 해 없음). 제약 조건을 확인하거나 완화해 보세요.")
    elif status == cp_model.MODEL_INVALID:
         update_status("[솔버 결과] 모델 정의에 오류가 있습니다.")

    # --- 결과 처리 ---
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        update_status(f"[결과 처리] 성공! 스케줄 데이터 추출 중..."); schedule = {}; solution_assignments = {}; extraction_ok = True
        try:
            solution_game_days = [solver.Value(game_day[g]) for g in range(num_games)]
            for g in range(num_games):
                day_num = solution_game_days[g] + 1
                schedule[g] = {'day': day_num, 'teams': [[] for _ in range(num_teams_per_game)], 'game_id': g + 1}
                for t in range(num_teams_per_game):
                    for p_idx in pos_indices:
                        key = (g, t, p_idx)
                        if key in assignment:
                            player_id = solver.Value(assignment[key])
                            solution_assignments[key] = player_id
                        else:
                             solution_assignments[key] = -1
                             extraction_ok = False
                             update_status(f"[결과 오류] 키 없음: g={g},t={t},p={p_idx}")

            if extraction_ok: update_status("[결과 처리] 스케줄 데이터 추출 완료.")
            else: pass

            return schedule, solution_assignments, status_messages, status
        except Exception as e:
             update_status(f"[결과 처리 오류] 값 추출 중 예외 발생: {e}")
             return None, None, status_messages, status

    else:
        update_status(f"[결과 처리] 실패.")
        return None, None, status_messages, status


# --- Streamlit UI (변경 없음) ---
st.title("🎮 선수 팀 배정 스케줄 생성기 (10 게임 고정)")
st.caption(f"총 {NUM_GAMES} 게임 ({NUM_DAYS}일 자동 분배, 1일 3~4게임) | 1티어 1회씩 맞대결 | 동포지션 적군 1회 고정 | 아군 조합 0회 매치업 최소화 | 적군 조합 0회 매치업 최소화") # <<< 캡션 수정

with st.sidebar:
    st.header("⚙️ 스케줄 생성 설정")
    time_limit_sec = st.slider(
        "계산 시간 (초)[높을수록 퀄리티 높은 대진]", min_value=10, max_value=300, value=10, step=10,
        help="솔버가 해를 찾는 최대 시간을 설정합니다. 시간이 짧으면 최적해를 찾지 못할 수 있습니다."
    )
    st.subheader("🚫 날짜별 출전 금지 선수")
    banned_players_by_day_ui = defaultdict(set)
    for d in range(1, NUM_DAYS + 1):
        multi_select_key = f"ban_day_{d}"
        default_banned_display = list(st.session_state.get(multi_select_key, []))
        banned_list_display = st.multiselect(f"{d}일차 출전 금지 선수", options=display_player_options, default=default_banned_display, key=multi_select_key)
        if banned_list_display:
            banned_players_by_day_ui[d] = set(banned_list_display)

st.header("🚀 스케줄 생성 실행")
results_area = st.container()

if st.button(f"{NUM_GAMES} 게임 스케줄 생성 시작!"):
    results_area.empty()
    with results_area:
        st.info(f"{NUM_GAMES} 게임 스케줄 생성 시도 (0회 매치업 최소화 목표, 최대 {time_limit_sec}초)...")
        overall_status_logs = []
        search_start_time = time.time(); solution_found = False
        final_schedule = None; final_assignments = None
        final_status = cp_model.UNKNOWN

        schedule_result, assignment_result, status_log, final_status = solve_schedule(
            positions=positions, player_data=player_data,
            num_teams_per_game=num_teams_per_game, players_per_team=players_per_team,
            banned_players_by_day=dict(banned_players_by_day_ui),
            time_limit_seconds=time_limit_sec
        )
        overall_status_logs.extend(status_log)

        search_end_time = time.time()
        st.info(f"실행 완료. (실제 소요 시간: {search_end_time - search_start_time:.2f}초 / 요청 시간 제한: {time_limit_sec}초)")

        if final_status == cp_model.OPTIMAL:
            st.success(f"성공! 최적 스케줄 발견!")
            final_schedule = schedule_result; final_assignments = assignment_result; solution_found = True
        elif final_status == cp_model.FEASIBLE:
            st.success(f"성공! 실행 가능한 스케줄 발견! (시간 제한 도달, 최적해가 아닐 수 있습니다)")
            final_schedule = schedule_result; final_assignments = assignment_result; solution_found = True
        elif final_status == cp_model.INFEASIBLE:
            st.error(f"실패: 제약 조건을 모두 만족하는 스케줄을 찾을 수 없습니다. (INFEASIBLE) **연속 경기 금지 조건이 너무 엄격할 수 있습니다.**") # <<< 메시지 추가
        elif final_status == cp_model.MODEL_INVALID:
             st.error(f"실패: 모델 정의에 오류가 있습니다. (MODEL_INVALID)")
        else:
             st.error(f"실패: 스케줄을 찾지 못했습니다. (상태: {cp_model.CpSolverStatus.Name(final_status)})")


        if solution_found and final_schedule and final_assignments:
            st.header(f"📊 최종 스케줄 ({NUM_GAMES} 게임)")
            schedule_table_data = []
            pos_indices = list(range(len(positions)))
            sorted_games = sorted(final_schedule.keys(), key=lambda g: (final_schedule[g]['day'], final_schedule[g]['game_id']))

            for g in sorted_games:
                game_info = final_schedule[g]
                game_data = {'Day': game_info['day'], 'Game': game_info['game_id'], 'vs': 'vs'}
                for t in range(num_teams_per_game):
                    team_prefix = 'Team A' if t == 0 else 'Team B'
                    for p_idx in pos_indices:
                        player_id = final_assignments.get((g, t, p_idx), -1)
                        display_text = "-"
                        p_pos = positions[p_idx]
                        if player_id != -1:
                            _, p_alias_found, p_pos_found, _ = get_player_info(player_id)
                            p_alias = p_alias_found if p_alias_found != "?" else "-"
                            display_text = str(p_alias) if p_alias else "-"
                            p_pos = p_pos_found if p_pos_found != "?" else p_pos

                        column_name = f"{team_prefix} ({p_pos})"
                        game_data[column_name] = display_text
                schedule_table_data.append(game_data)

            if schedule_table_data:
                schedule_df = pd.DataFrame(schedule_table_data)
                schedule_df = schedule_df.sort_values(by=['Day', 'Game'])

                display_rows = []
                last_day = None
                column_order = ['Day']
                team_a_cols = [f"Team A ({pos})" for pos in positions]
                team_b_cols = [f"Team B ({pos})" for pos in positions]
                column_order.extend(team_a_cols)
                column_order.append('vs')
                column_order.extend(team_b_cols)
                blank_row_dict = {col: '' for col in column_order}

                for index, row in schedule_df.iterrows():
                    current_day = row['Day']
                    if last_day is not None and current_day != last_day:
                        display_rows.append(blank_row_dict.copy())

                    display_row_data = {col: row.get(col, '') for col in column_order}
                    display_rows.append(display_row_data)
                    last_day = current_day

                display_df = pd.DataFrame(display_rows, columns=column_order)
                st.dataframe(display_df.style.hide(axis="index"), use_container_width=True)

            else:
                 st.warning("스케줄 데이터 생성 중 문제가 발생했습니다.")


        with st.expander("상세 실행 로그 보기"):
            st.text("\n".join(overall_status_logs))