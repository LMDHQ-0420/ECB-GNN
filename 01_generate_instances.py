import pyscipopt
import json
from pyscipopt import quicksum
import os
import argparse
import shutil
import copy
import random


def read_constant_json(constant_file_path):
    with open(constant_file_path, "r") as file:
        constant = json.load(file)

    generators = constant.get("Generators", [])
    transmission_lines = constant.get("Transmission lines", [])
    contingencies = constant.get("Contingencies", [])
    buses = constant.get("Buses", [])
    renewable_energy = constant.get("Renewable Energy", [])
    storage_units = constant.get("Storage units", [])

    for bus_key in buses.keys():
        load_value = buses[bus_key]["Load (MW)"]
        if not isinstance(load_value, list):
            buses[bus_key]["Load (MW)"] = [load_value] * 744

    return generators, transmission_lines, contingencies, buses, renewable_energy, storage_units


def create_model(generators, buses, renewable_energy, storage_units, time_steps, begin_time, lp_file_path):

    model = pyscipopt.Model("power grid dispatching")

    M = 100000

    p_G = [[model.addVar(vtype="C", name=f"p_G_{name_G}_{t}", lb=0)
            for t in range(time_steps + 1)] for name_G in generators]
    delta_p_G_k = [[[model.addVar(vtype="C", name=f"delta_p_G_k_{name_G}_{t}_{k}", lb=0)
                     for k in range(4)] for t in range(time_steps + 1)] for name_G in generators]
    p_R = [[model.addVar(vtype="C", name=f"p_R_{name_R}_{t}", lb=0)
            for t in range(time_steps + 1)] for name_R in renewable_energy]
    delta_p_R = [[model.addVar(vtype="C", name=f"delta_p_R_{name_R}_{t}", lb=0)
                     for t in range(time_steps + 1)] for name_R in renewable_energy]
    SoC_S = [[model.addVar(vtype="C", name=f"SoC_S_{name_S}_{t}", lb=0)
              for t in range(time_steps + 1)] for name_S in storage_units]
    delta_SoC_S_CHARGE = [[model.addVar(vtype="C", name=f"delta_SoC_S_CHARGE_{name_S}_{t}", lb=0)
                           for t in range(time_steps + 1)] for name_S in storage_units]
    delta_SoC_S_DISCHARGE = [[model.addVar(vtype="C", name=f"delta_SoC_S_DISCHARGE_{name_S}_{t}", lb=0)
                              for t in range(time_steps + 1)] for name_S in storage_units]

    y_G = [[model.addVar(vtype="I", name=f"y_G_{name_G}_{t}", lb=0, ub=1)
            for t in range(time_steps + 1)] for name_G in generators]
    tier_G_k = [[[model.addVar(vtype="I", name=f"tier_G_k_{name_G}_{t}_{k}", lb=0, ub=1)
            for k in range(4)] for t in range(time_steps + 1)] for name_G in generators]
    y_R = [[model.addVar(vtype="I", name=f"y_R_{name_R}_{t}", lb=0, ub=1)
              for t in range(time_steps + 1)] for name_R in renewable_energy]
    y_S = [[model.addVar(vtype="I", name=f"y_S_{name_S}_{t}", lb=0, ub=1)
              for t in range(time_steps + 1)] for name_S in storage_units]


    for index_G, (name_G, info_G) in enumerate(generators.items()):
        initial_generators_P = info_G["Initial power (MW)"]
        model.fixVar(p_G[index_G][0], initial_generators_P)
        initial_generators_Y = 1 if info_G["Initial status (h)"] > 0 else 0
        model.fixVar(y_G[index_G][0], initial_generators_Y)

    for index_G, (name_R, info_R) in enumerate(renewable_energy.items()):
        model.fixVar(p_R[index_G][0], 0)

    for index_G, (name_S, info_S) in enumerate(storage_units.items()):
        initial_storage_units_SOC = info_S["Initial level (MWh)"]
        model.fixVar(SoC_S[index_G][0], initial_storage_units_SOC)

    c_G_k = [[(info_G["Production cost curve ($)"][k + 1] - info_G["Production cost curve ($)"][k])
                         / (info_G["Production cost curve (MW)"][k + 1] - info_G["Production cost curve (MW)"][k])
                        for k in range(4)] for index_G, (name_G, info_G) in enumerate(generators.items())]

    all_cost = 0.0

    for time in range(1, time_steps + 1):

        model.addCons(quicksum(p_G[i][time] for i in range(len(generators)))
                      + quicksum(p_R[i][time] for i in range(len(renewable_energy)))
                      - quicksum(SoC_S[i][time] - SoC_S[i][time - 1] for i in range(len(storage_units)))
                      == quicksum(info_B["Load (MW)"][begin_time + time - 1] for index_B, (name_B, info_B) in enumerate(buses.items())),
                      name=f"SystemPowerBalence_{time}")


        for index_G, (name_G, info_G) in enumerate(generators.items()):

            model.addCons(p_G[index_G][time] >= info_G["Production cost curve (MW)"][0] * y_G[index_G][time],
                          name=f"G_MinPowerLimit_{name_G}_{time}")

            model.addCons(p_G[index_G][time] <= info_G["Production cost curve (MW)"][4] * y_G[index_G][time],
                          name=f"G_MaxPowerLimit_{name_G}_{time}")

            model.addCons(quicksum([p_G[index_G][time], -p_G[index_G][time - 1]])
                          <= quicksum([info_G['Ramp up limit (MW)'] * y_G[index_G][time - 1],
                                       info_G['Startup limit (MW)'] * (1 - y_G[index_G][time - 1])]),
                          name=f"G_PowerUpLimit_{name_G}_{time}")

            model.addCons(quicksum([p_G[index_G][time - 1], -p_G[index_G][time]])
                          <= quicksum([info_G['Ramp down limit (MW)'] * y_G[index_G][time],
                                       info_G['Shutdown limit (MW)'] * (1 - y_G[index_G][time])]),
                          name=f"G_PowerDownLimit_{name_G}_{time}")

            model.addCons(quicksum([p_G[index_G][time], - info_G["Production cost curve (MW)"][0]]) >= - M * (1 - y_G[index_G][time]),
                          name=f"G_RunningFlagSolve1_{name_G}_{time}")
            model.addCons(quicksum([p_G[index_G][time], - info_G["Production cost curve (MW)"][0]]) <= M * y_G[index_G][time],
                          name=f"G_RunningFlagSolve2_{name_G}_{time}")

            model.addCons(quicksum(tier_G_k[index_G][time][c] for c in range(4)) == y_G[index_G][time],
                          name=f"G_PowerCurveOnlyLimit_{name_G}_{time}")

            model.addCons(p_G[index_G][time] >=
                          quicksum(tier_G_k[index_G][time][c] * info_G["Production cost curve (MW)"][c] for c in range(4)),
                          name=f"G_PowerCurveSolve1_{name_G}_{time}")
            model.addCons(p_G[index_G][time] <=
                          quicksum(tier_G_k[index_G][time][c] * info_G["Production cost curve (MW)"][c + 1]for c in range(4)),
                          name=f"G_PowerCurveSolve2_{name_G}_{time}")

            for k in range(4):


                model.addCons(delta_p_G_k[index_G][time][k] <= M * tier_G_k[index_G][time][k],
                              name=f"G_DeltaPowerOnlyLimit_{name_G}_{time}_{k}")

                model.addCons(delta_p_G_k[index_G][time][k]
                              >= p_G[index_G][time] -info_G["Production cost curve (MW)"][k]
                                           - M * (1- tier_G_k[index_G][time][k]),
                              name=f"G_DeltaPowerSolve_{name_G}_{time}_{k}")

                all_cost += info_G["Production cost curve ($)"][k] * tier_G_k[index_G][time][k]
                all_cost += delta_p_G_k[index_G][time][k] * c_G_k[index_G][k]


        for index_S, (name_S, info_S) in enumerate(storage_units.items()):

            delta_SoC_t = quicksum([SoC_S[index_S][time], -SoC_S[index_S][time - 1]])

            model.addCons(SoC_S[index_S][time] <= info_S["Maximum level (MWh)"],
                          name=f"S_MaxSoCLimit_{name_S}_{time}")

            model.addCons(SoC_S[index_S][time] >= info_S["Minimum level (MWh)"],
                          name=f"S_MinSoCLimit_{name_S}_{time}")

            model.addCons(delta_SoC_t <= info_S["Maximum charge rate (MW)"],
                          name=f"S_ChargeLimit_{name_S}_{time}")

            model.addCons(- delta_SoC_t <= info_S["Maximum discharge rate (MW)"],
                name=f"S_DischargeLimit_{name_S}_{time}")

            model.addCons(delta_SoC_t >= - M * (1 - y_S[index_S][time]),
                          name=f"S_ChargeFlagSolve1_{name_S}_{time}")
            model.addCons(delta_SoC_t <= M * y_S[index_S][time],
                          name=f"S_ChargeFlagSolve2_{name_S}_{time}")

            model.addCons(delta_SoC_S_CHARGE[index_S][time] >= delta_SoC_t,
                          name=f"S_ChargeSoCMinSolve_{name_S}_{time}")
            model.addCons(delta_SoC_S_DISCHARGE[index_S][time] >= - delta_SoC_t,
                          name=f"S_DischargeSoCMinSolve_{name_S}_{time}")

            all_cost += quicksum([delta_SoC_S_CHARGE[index_S][time] * info_S["Charge cost ($/MW)"],
                                  delta_SoC_S_DISCHARGE[index_S][time] * info_S["Discharge cost ($/MW)"]])


        for index_R, (name_R, info_R) in enumerate(renewable_energy.items()):

            delta_P_R_t = quicksum([p_R[index_R][time], -p_R[index_R][time - 1]])

            model.addCons(delta_P_R_t >= - M * (1 - y_R[index_R][time]),
                          name=f"R_PowerUpFlagSolve1_{name_R}_{time}")
            model.addCons(delta_P_R_t <= M * y_R[index_R][time],
                          name=f"R_PowerDownFlagSolve2_{name_R}_{time}")

            model.addCons(delta_p_R[index_R][time] >= delta_P_R_t,
                          name=f"R_DeltaPowerMinSolve1_{name_R}_{time}")
            model.addCons(delta_p_R[index_R][time] >= - delta_P_R_t,
                          name=f"R_DeltaPowerMinSolve2_{name_R}_{time}")

            all_cost += delta_p_R[index_R][time] * info_R["Curtail Cost ($/MW)"]


    model.setObjective(all_cost, sense='minimize')

    if not os.path.exists(os.path.dirname(lp_file_path)):
        os.makedirs(os.path.dirname(lp_file_path))
    model.writeProblem(lp_file_path)
    return


def perturb_constants(out_dir, prefix, components, diff, tstep, i, seed, scale=0.05, begin_time=0):
    random.seed(seed)
    gens = copy.deepcopy(components[0])
    trans = copy.deepcopy(components[1])
    cont = copy.deepcopy(components[2])
    buses = copy.deepcopy(components[3])
    rens = copy.deepcopy(components[4])
    stors = copy.deepcopy(components[5])

    for bus in buses.values():
        bus["Load (MW)"] = [v * random.uniform(1-scale, 1+scale) for v in bus["Load (MW)"]]

    lp_file_path = f"{out_dir}/{prefix}_perturb{scale}_{diff}_{i}.lp"
    create_model(gens, buses, rens, stors, tstep, i % 24 * 24, lp_file_path)

def structural_variation(out_dir, prefix, components, diff, tstep, i, seed, frac=0.1, begin_time=0):
    random.seed(seed)
    
    n_gens = len(components[0])
    n_stors = len(components[5])
    n_rens = len(components[4])
    
    gen_var = max(1, int(round(n_gens * frac)))
    stor_var = max(1, int(round(n_stors * frac)))
    ren_var = max(1, int(round(n_rens * frac)))
    
    gen_add = random.randint(1, gen_var)
    gen_remove = random.randint(0, min(gen_var, n_gens-1))
    
    storage_add = random.randint(1, stor_var) if n_stors > 0 else 0
    storage_remove = random.randint(0, min(stor_var, n_stors-1)) if n_stors > 0 else 0
    
    ren_add = random.randint(1, ren_var) if n_rens > 0 else 0
    ren_remove = random.randint(0, min(ren_var, n_rens-1)) if n_rens > 0 else 0

    gens = copy.deepcopy(components[0])
    trans = copy.deepcopy(components[1])
    cont = copy.deepcopy(components[2])
    buses = copy.deepcopy(components[3])
    rens = copy.deepcopy(components[4])
    stors = copy.deepcopy(components[5])

    gens_list = list(gens.items())
    if gen_remove > 0 and len(gens_list) > gen_remove:
        keys_to_remove = random.sample([k for k, _ in gens_list], gen_remove)
        for k in keys_to_remove:
            gens.pop(k)
    
    if gen_add > 0 and len(gens_list) > 0:
        for i_add in range(gen_add):
            template_key, template_value = random.choice(gens_list)
            new_g = copy.deepcopy(template_value)
            new_name = f"Gen_added_{i}_{i_add}"
            new_g["Name"] = new_name
            gens[new_name] = new_g

    stors_list = list(stors.items())
    if storage_remove > 0 and len(stors_list) > storage_remove:
        keys_to_remove = random.sample([k for k, _ in stors_list], storage_remove)
        for k in keys_to_remove:
            stors.pop(k)
    
    if storage_add > 0 and len(stors_list) > 0:
        for i_add in range(storage_add):
            template_key, template_value = random.choice(stors_list)
            new_s = copy.deepcopy(template_value)
            new_name = f"Storage_added_{i}_{i_add}"
            new_s["Name"] = new_name
            stors[new_name] = new_s

    rens_list = list(rens.items())
    if ren_remove > 0 and len(rens_list) > ren_remove:
        keys_to_remove = random.sample([k for k, _ in rens_list], ren_remove)
        for k in keys_to_remove:
            rens.pop(k)
    
    if ren_add > 0 and len(rens_list) > 0:
        for i_add in range(ren_add):
            template_key, template_value = random.choice(rens_list)
            new_r = copy.deepcopy(template_value)
            new_name = f"Ren_added_{i}_{i_add}"
            new_r["Name"] = new_name
            rens[new_name] = new_r
    
    lp_file_path = f"{out_dir}/{prefix}_structvar_{diff}_{i}.lp"
    create_model(gens, buses, rens, stors, tstep, i % 24 * 24, lp_file_path)



if '__main__' == __name__:
    parser = argparse.ArgumentParser(description="Generate the LP dataset")
    parser.add_argument("--mode", choices=["train", "test", "valid"], required=True, help="Please select to generate the training set, test set or validation set.")
    args = parser.parse_args()

    components = read_constant_json("data/constant.json")

    n_gens = len(components[0])
    n_stors = len(components[5])
    n_rens = len(components[4])

    difficulties = [("simple", 24), ("middle", 72), ("hard", 120)]
    n_per_difficulty = 100 if args.mode == "train" else 25

    if args.mode == "train":
        out_dir = "data/instances/train"
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        seed = 0
        for diff, tstep in difficulties:
            for i in range(n_per_difficulty):
                perturb_constants(out_dir, "train", components, diff, tstep, i,
                                           seed=seed+1, scale=0.05)
                perturb_constants(out_dir, "train", components, diff, tstep, i,
                                             seed=seed+1, scale=0.2)
                structural_variation(out_dir, "train", components, diff, tstep, i, seed=seed+1)

    elif args.mode == "valid":
        out_dir = "data/instances/valid"
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        seed = 100000
        for diff, tstep in difficulties:
            for i in range(n_per_difficulty):
                perturb_constants(out_dir, "valid", components, diff, tstep, i,
                                           seed=seed+1, scale=0.05)
                perturb_constants(out_dir, "valid", components, diff, tstep, i,
                                             seed=seed+1, scale=0.15)
                structural_variation(out_dir, "valid", components, diff, tstep, i, seed=seed+1)

    elif args.mode == "test":
        out_dir = "data/instances/test"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        seed = 200000
        for diff, tstep in difficulties:
            subdir = f"{out_dir}/ParamRobustness/{diff}"
            os.makedirs(subdir, exist_ok=True)
            for i in range(n_per_difficulty):
                perturb_constants(subdir, "test_paramrobust", components, diff, tstep, i,
                                seed=seed+1, scale=0.15)
        for diff, tstep in difficulties:
            subdir = f"{out_dir}/StructTransfer/{diff}"
            os.makedirs(subdir, exist_ok=True)
            for i in range(n_per_difficulty):
                structural_variation(subdir, "test_structtransfer", components, diff, tstep, i,
                                              seed=seed+1, frac=0.2)
        for diff, tstep in difficulties:
            subdir = f"{out_dir}/Core/{diff}"
            os.makedirs(subdir, exist_ok=True)
            for i in range(n_per_difficulty):
                lp_file_path = f"{subdir}/test_core_{diff}_{i}.lp"
                create_model(components[0], components[3], components[4], components[5], tstep, i*24, lp_file_path)