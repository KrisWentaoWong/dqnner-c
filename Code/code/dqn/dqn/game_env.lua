

local env = torch.class('GameEnv')      -- lutm: create a new torch class named 'GameEnv'.

local json = require ("dkjson")
local zmq = require "lzmq"

if pcall(require, 'signal') then
    signal.signal("SIGPIPE", function() print("raised") end)
else
    print("No signal module found. Assuming SIGPIPE is okay.")
end

function env:__init(args)               -- lutm: I think it's the constructor for the class 'env'.

    self.ctx = zmq.context()
    self.skt = self.ctx:socket{zmq.REQ, --- lutm: skt is short for socket.
        linger = 0, rcvtimeo = 10000;
        connect = "tcp://127.0.0.1:" .. args.zmq_port;
    }

    if args.mode == 'NER_C' then      --- lutm: so, this file is written by MIT, but not DeepMind
        self.actions = {1,2}
    else
        self.actions = {0,1,2,3,4}
    end
    assert(args.mode=='NER_C', "args.mode=" .. args.mode)
end

function env:process_msg(msg)    
    -- screen, reward, terminal
    -- print("MESSAGE:", msg)
    loadstring(msg)()                   --- lutm: Like 'eval(...)' in JavaScript, this function load a string as 'code' and run it.
    -- if reward ~= 0 then              ---       So, the parameter msg is a command, a function call, or a script.
    --     print('non-zero reward', reward)
    -- end
    --- msg是server.py发来的，内容是一个字符串，对s, r, terminal赋值
    --- 也就是说，
    return torch.Tensor(state), reward, terminal, stepSeqForCurrentMention, isNullByLastQuery, isHumanByLastQuery, hasQueriedHuman, currentLR
end

function env:newGame()
    self.skt:send("newGame")
    msg = self.skt:recv()
    while msg == nil do
        msg = self.skt:recv()
    end
    return self:process_msg(msg)
end


function env:step(action, query)
    -- assert(action==1 or action==0, "Action " .. tostring(action))
    self.skt:send(tostring(action) .. ' ' .. tostring(query))
    msg = self.skt:recv()
    while msg == nil do
        msg = self.skt:recv()
    end
    return self:process_msg(msg)
end

function env:testStart()
    self.skt:send("testStart")
    msg = self.skt:recv()
    assert(msg == 'done', msg)
end

function env:testEnd()
    self.skt:send("testEnd")
    msg = self.skt:recv()
    assert(msg == 'done' or msg=="earlyStop", msg)
    if msg=="earlyStop" then
        print('\n\n******************************** earlyStop\n\n')
        return 1
    else
        return 0
    end
end

function env:devStart()
    self.skt:send("devStart")
    msg = self.skt:recv()
    assert(msg == 'done', msg)
end

function env:devEnd()
    self.skt:send("devEnd")
    msg = self.skt:recv()
    assert(msg == 'done', msg)
end


function env:getActions()   
    return self.actions

end
