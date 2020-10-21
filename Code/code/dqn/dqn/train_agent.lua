require 'xlua'      ---lutm: This script is called in shell run_cpu, and run_gpu.sh.
require 'optim'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-epoches', 100, 'number of epoches')
cmd:option('-training_episodes', 100, 'number of training episodes')
cmd:option('-eval_episodes', 100, 'number of evaluation episodes')
cmd:option('-dev_episodes', 100, 'number of evaluation episodes')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')

cmd:option('-zmq_port', 5050, 'ZMQ port')
cmd:option('-mode', 'Shooter', 'Experiment domain')
cmd:option('-exp_folder', 'logs/', 'folder for logs')

cmd:text()

local opt = cmd:parse(arg)  -- lutm: opt stands for option.
print("opt.epoches=", opt.epoches)
print("opt.training_episodes=", opt.training_episodes)

if not dqn then             -- lutm: In the beginning, dqn is null.
    require "initenv"
end

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)       --- lutm: setup函数定义在initenv.lua中

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local globalSteps = 0
time_history[1] = 0

local total_reward
local nrewards
local n_finished_eval_episodes
local eval_steps
local episode_reward
local n_finished_dev_episodes
local dev_steps

print("Iteration ..", globalSteps)
local win = nil

---- LOOP OF EPOCHES (TRAINING)
local finished_training_epoches = 0
local earlyStop = 0
while earlyStop==0 and finished_training_epoches < opt.epoches do           -- lutm: 对步数循环
    ---- START OF AN EPOCH (TRAINING)
    local finished_episodes_in_epoch = 0
    while finished_episodes_in_epoch < opt.training_episodes do
        ---- START OF AN EPISODE (TRAINING)
        local state, reward, terminal, stepSeqForCurrentMention, isNullByLastQuery, isHumanByLastQuery, hasQueriedHuman, currentLR = game_env:newGame()
        if currentLR>0 then agent:setLRFromServer(currentLR) end

        local steps_in_episode = 1
        while true do
            globalSteps = globalSteps + 1
            if globalSteps % 100 == 0 then
                xlua.progress(globalSteps, opt.steps)
            end

            steps_in_episode = steps_in_episode + 1
            if terminal then
                ---- end of current episode
                finished_episodes_in_epoch = finished_episodes_in_epoch + 1
                break
            end

            ---- get actions from agent
            local action_index, query_index = agent:perceive(reward, state, terminal, stepSeqForCurrentMention, isNullByLastQuery, isHumanByLastQuery, hasQueriedHuman)

            ---- send actions to env
            state, reward, terminal, stepSeqForCurrentMention, isNullByLastQuery, isHumanByLastQuery, hasQueriedHuman = game_env:step(game_actions[action_index], query_index)

            if globalSteps %1000 == 0 then collectgarbage() end     -- lutm: to finalize unused objects.

            if globalSteps %1000 == 0 then
                print(" currentLR: ", currentLR, "Epoches: ", (finished_training_epoches+1), " Episodes: ", finished_episodes_in_epoch, " Steps: ", steps_in_episode, " GlobalSteps: ", globalSteps)
                --agent:report()
            end
        end

        if finished_episodes_in_epoch == 1 or finished_episodes_in_epoch%10==0 then
            print(" currentLR: ", currentLR, "Epoches: ", (finished_training_epoches+1), " Episodes: ", finished_episodes_in_epoch, " Steps: ", steps_in_episode, " GlobalSteps: ", globalSteps)
            --agent:report()
        end
    end
    finished_training_epoches = finished_training_epoches + 1
    ---- END OF AN EPOCH (TRAINING)

    ---------------------------------------------------------------------- EVAL START
    if globalSteps > learn_start then

        -- **************************
        -- **************************
        game_env:devStart()
        local state, reward, terminal, stepSeqForCurrentMention, isNullByLastQuery, isHumanByLastQuery, hasQueriedHuman, currentLR = game_env:newGame()

        -- 整个测试集上，已完成的情节数
        n_finished_dev_episodes = 0
        dev_steps = 0

        print("Deving...")

        while true do   --- dev_steps=5000
            if dev_steps%100==0 then
                xlua.progress(n_finished_dev_episodes, opt.dev_episodes)
            end
            dev_steps = dev_steps + 1

            --- testing=true，不进行抽样、记忆等操作
            --- epsilong=0.0，不探索，直接greedy
            local action_index, query_index = agent:perceive(reward, state, terminal, stepSeqForCurrentMention, isNullByLastQuery, isHumanByLastQuery, hasQueriedHuman, true, 0.0)

            -- Play game in test mode 
            state, reward, terminal, stepSeqForCurrentMention, isNullByLastQuery, isHumanByLastQuery, hasQueriedHuman = game_env:step(game_actions[action_index], query_index)

            if dev_steps%1000 == 0 then collectgarbage() end

            if terminal then
                n_finished_dev_episodes = n_finished_dev_episodes + 1
                state, reward, terminal, stepSeqForCurrentMention, isNullByLastQuery, isHumanByLastQuery, hasQueriedHuman = game_env:newGame()

                if n_finished_dev_episodes == opt.dev_episodes then
                    break
                end
            end
        end

        game_env:devEnd()

        --****************************************
        --****************************************

        game_env:testStart()
        local state, reward, terminal, stepSeqForCurrentMention, isNullByLastQuery, isHumanByLastQuery, hasQueriedHuman, currentLR = game_env:newGame()

        -- test_avg_Q = test_avg_Q or optim.Logger(paths.concat(opt.exp_folder , 'test_avgQ.log'))
        test_avg_R = test_avg_R or optim.Logger(paths.concat(opt.exp_folder , 'test_avgR.log'))

        -- 整个测试集上的奖励
        total_reward = 0
        -- 整个测试集上的奖励数（小于等于step数）
        nrewards = 0
        -- 整个测试集上，已完成的情节数
        n_finished_eval_episodes = 0
        -- each情节的累计奖励
        episode_reward = 0
        -- 已评估的步数
        eval_steps = 0

        local eval_time = sys.clock()
        print("Testing...")

        while true do   --- eval_steps=5000
            if eval_steps%100==0 then
                xlua.progress(n_finished_eval_episodes, opt.eval_episodes)
            end
            eval_steps = eval_steps + 1

            --- testing=true，不进行抽样、记忆等操作
            --- epsilong=0.0，不探索，直接greedy
            local action_index, query_index = agent:perceive(reward, state, terminal, stepSeqForCurrentMention, isNullByLastQuery, isHumanByLastQuery, hasQueriedHuman, true, 0.0)

            -- Play game in test mode
            state, reward, terminal, stepSeqForCurrentMention, isNullByLastQuery, isHumanByLastQuery, hasQueriedHuman = game_env:step(game_actions[action_index], query_index)

            if eval_steps%1000 == 0 then collectgarbage() end

            -- record every reward
            episode_reward = episode_reward + reward
            if reward ~= 0 then
               nrewards = nrewards + 1
            end

            if terminal then
                total_reward = total_reward + episode_reward
                episode_reward = 0
                n_finished_eval_episodes = n_finished_eval_episodes + 1
                state, reward, terminal, stepSeqForCurrentMention, isNullByLastQuery, isHumanByLastQuery, hasQueriedHuman = game_env:newGame()

                if n_finished_eval_episodes == opt.eval_episodes then
                    break
                end
            end
        end

        earlyStop = game_env:testEnd()

        state, reward, terminal, stepSeqForCurrentMention, isNullByLastQuery, isHumanByLastQuery, hasQueriedHuman = game_env:newGame() --start new game

        eval_time = sys.clock() - eval_time
        start_time = start_time + eval_time
        --20190917 error
--        agent:compute_validation_statistics()
        local ind = #reward_history+1
        -- 取平均数，转换为每个情节的累计奖励
        total_reward = total_reward/math.max(1, n_finished_eval_episodes)

        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
            agent.best_network = agent.network:clone()
        end

        if agent.v_avg then
            v_history[ind] = agent.v_avg
            td_history[ind] = agent.tderr_avg
            qmax_history[ind] = agent.q_max
        end
        print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

        -- plotting graphs
        test_avg_R:add{['Average Reward'] = total_reward}
        -- test_avg_Q:add{['Average Q'] = agent.v_avg}
     
        test_avg_R:style{['Average Reward'] = '-'}; test_avg_R:plot()        
        -- test_avg_Q:style{['Average Q'] = '-'}; test_avg_Q:plot()

        reward_history[ind] = total_reward
        reward_counts[ind] = nrewards
        episode_counts[ind] = n_finished_eval_episodes

        time_history[ind+1] = sys.clock() - start_time

        local time_dif = time_history[ind+1] - time_history[ind]

        local training_rate = opt.actrep*opt.eval_freq/time_dif

        print(string.format(
            '\nEpochs: %d, lr: %G, Steps: %d reward: %.2f, epsilon: %.2f, ' ..
            'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
            'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d',
            n_finished_eval_episodes, agent.lr, globalSteps, globalSteps *opt.actrep, total_reward, agent.ep, time_dif,
            training_rate, eval_time, opt.actrep*eval_steps/eval_time,
            nrewards))
    end
    ---------------------------------------------------------------------- EVAL END

    if globalSteps % opt.save_freq == 0 or globalSteps == opt.steps then
        local s, a, o, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_o,
        agent.valid_r, agent.valid_s2, agent.valid_term
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = nil, nil, nil, nil, nil, nil, nil
        local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
            agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

        local filename = opt.name
        if opt.save_versions > 0 then
            filename = filename .. "_" .. math.floor(globalSteps / opt.save_versions)
        end
        filename = filename
        torch.save(opt.exp_folder .. filename .. ".t7", {agent = agent,
                                model = agent.network,
                                best_model = agent.best_network,
                                reward_history = reward_history,
                                reward_counts = reward_counts,
                                episode_counts = episode_counts,
                                time_history = time_history,
                                v_history = v_history,
                                td_history = td_history,
                                qmax_history = qmax_history,
                                arguments=opt})
        if opt.saveNetworkParams then
            local nets = {network=w:clone():float()}
            torch.save(opt.exp_folder .. filename..'.params.t7', nets, 'ascii')
        end
        agent.valid_s, agent.valid_a, agent.valid_o, agent.valid_r, agent.valid_s2,
            agent.valid_term = s, a, o, r, s2, term
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
        print('Saved:', opt.exp_folder .. filename .. '.t7')
        io.flush()
        collectgarbage()
    end

    if earlyStop==1 then
        print("*****  earlyStop")
        break
    end
end