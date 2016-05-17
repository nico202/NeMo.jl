module NeMo

export initnemo
export delconf, stdoutlog, synapseswo
export delnet, addizneuron, addsynapse
export ncount, walltime, simtime, timereset
export newsim, delsim, simstep
export getmembranes

## TODO: fix this
const libnemo = "libnemo"

## Configuration
"Create a configuration pointer"
newconf() = ccall((:nemo_new_configuration, libnemo), Ptr{UInt8}, ())

## FIXME: prevent segfault on multiple calls
"Delete a configuration. Segfault if called on unexisting configuration"
function delconf(conf)
    ccall((:nemo_delete_configuration, libnemo),
                  Void, (Ptr{UInt8},), conf)
end

"Enables NeMosim log"
stdoutlog(conf) = ccall((:nemo_log_stdout, libnemo), Ptr{UInt8}, (Ptr{UInt8},), conf) ## Returns null?
#TODO: add cpu/gpu conf

"Set the synapses Write-Only (disable in-simulation read). Faster net creation"
synapseswo(conf) = ccall((:nemo_set_write_only_synapses, libnemo), Ptr{UInt8}, (Ptr{UInt8},), conf)

## Construction
"Create a network (returns pointer)"
newnet() = ccall((:nemo_new_network, libnemo), Ptr{UInt8}, ())
## FIXME: prevent segfault on multiple calls
"Delete a network. Segfault on non-existing"
delnet(net) = ccall((:nemo_delete_network, libnemo), Void, (Ptr{UInt8},), (net));

## TODO: write
## function addntype(net; name = "Izhikevich")
##     ntype = Ref{Cuint}(0)
##     ccall((:nemo_add_neuron_type, libnemo), Ptr{UInt8}, (Ptr{UInt8}, Cstring, Ref{Cuint}), net, name, ntype)
##     ntype
## end

"Add an Izhikevich neuron to the net
args: net, index, iz-params ..."
function addizneuron(net, idx, a, b, c, d, u, v, sigma)
    res = ccall((:nemo_add_neuron_iz, libnemo), Ptr{UInt8}, (Ptr{UInt8}, UInt32, Float32, Float32, Float32, Float32, Float32, Float32, Float32), net, idx, a, b, c, d, u, v, sigma)
    if res != C_NULL
        error("You have added 2 neurons with the same id! Exiting")
    end
    res
end

function addsynapse(net, source, target, delay, weight, isplastic::Bool)
    sid = Ref{UInt8}(0)
    ccall((:nemo_add_synapse, libnemo), Ptr{UInt8}, (Ptr{UInt8}, UInt32, UInt32, UInt32, Float32, UInt8, Ptr{UInt8}), net, source, target, delay, weight, isplastic, sid)
    Int(sid[])
end

function ncount(net)
    count = Ref{Cuint}(0)
    ccall((:nemo_neuron_count, libnemo), Ptr{UInt8}, (Ptr{UInt8}, Ref{Cuint}), net, count);
    Int(count[])
end

## Simulation

newsim(net, conf) = ccall((:nemo_new_simulation, libnemo), Ptr{UInt8}, (Ptr{UInt8}, Ptr{UInt8}), net, conf)

delsim(sim) = ccall((:nemo_delete_simulation, libnemo), Void, (Ptr{UInt8},), sim)

function simstep(sim, neurons, fstim::Vector{UInt32}, istim::Vector{UInt32}, Iistim::Vector{Float32})
    nfired = Ref{Cuint}(0)
    fired::Ref{Vector{Cuint}}
    fired = zeros(neurons)
    istim::Array{UInt32}
    fstim::Array{UInt32}
    Iistim::Array{Float32}
    res = ccall((:nemo_step, libnemo)
                , Ptr{UInt8}
                , (Ptr{UInt8}
                   , Ptr{Cuint}, Csize_t
                   , Ptr{Cuint}, Ptr{Cfloat}, Csize_t
                   , Ptr{Vector{Cuint}}, Ptr{Cuint})
                , sim
                , convert(Ptr{Cuint}, pointer(fstim)), length(fstim)
                , convert(Ptr{Cuint}, pointer(istim)), convert(Ptr{Cuint}, pointer(Iistim)), length(istim)
                , fired, nfired)
    res != C_NULL ? error(errorname()) : fired[][1:nfired[]]
end

const fstim = Array{UInt32}[] ## Probably never used, speedup
const lfstim = length(fstim)
const pointfstim = convert(Ptr{Cuint}, pointer(fstim))
"Overloaded simstep with static fstims. ~3% faster"
function simstep(sim, neurons, istim::Vector{UInt32}, Iistim::Vector{Float32})
    nfired = Ref{Cuint}(0)
    fired::Ref{Vector{Cuint}} = zeros(neurons)
    res = ccall((:nemo_step, libnemo)
                , Ptr{UInt8}
                , (Ptr{UInt8}
                   , Ptr{Cuint}, Csize_t
                   , Ptr{Cuint}, Ptr{Cfloat}, Csize_t
                   , Ptr{Vector{Cuint}}, Ptr{Cuint})
                , sim
                , pointfstim, lfstim
                , convert(Ptr{Cuint}, pointer(istim)), convert(Ptr{Cuint}, pointer(Iistim)), length(istim)
                , fired, nfired)
    res != C_NULL ? error(errorname()) : fired[][1:nfired[]]
end


## Querying the network
"Returns the memrane value of a single neuron (selected by its index)"
function getmembrane(sim::Ptr{UInt8}, idx::UInt32)
    v = Ref{Float32}(0)
    ccall((:nemo_get_membrane_potential, libnemo), Ptr{UInt8},
          (Ptr{UInt8}, Cuint, Ref{Float32}), sim, idx, v)
    v[]
end
"Returns a list of tuples with (idx, potential)
for all membrane valus of the neuron in a given indexes array.
If you don't want automatic indexing, use getmembranes2"
getmembranes(sim::Ptr{UInt8}, idxs::Array{UInt32}) =
    [ (id, getmembrane(sim, id)) for id in idxs]
"Same as getmembranes, but returns just a list
(then ie. enumerate() it)"
getmembranes2(sim::Ptr{UInt8}, idxs::Array{Int}) =
    [ getmembrane(sim, id) for id in idxs]
## How to use vectorize?
## @vectorize_2arg Real getmembrane;

## Macro?
"Wall time elapsed from start/reset"
function walltime(sim)
    time = Ref{UInt}(0)
    ccall((:nemo_elapsed_wallclock, libnemo), Ptr{UInt8},
          (Ptr{UInt8}, Ref{UInt}), sim, time)
    UInt(time[])
end

"Simulation time elapsed"
function simtime(sim)
    time = Ref{UInt}(0)
    ccall((:nemo_elapsed_simulation, libnemo), Ptr{UInt8},
          (Ptr{UInt8}, Ref{UInt}), sim, time)
    UInt(time[])
end
"Reset the time count"
timereset() = ccall((:nemo_reset_timer, libnemo), Ptr{UInt8}, ())

## Error handling
errorname() = bytestring(ccall((:nemo_strerror, libnemo), Ptr{UInt8}, ()))

#######################################################################
"Public. Initialize nemo, returns network, config"
initnemo() = newnet(), newconf()

"Private: add 1 neuron to net"
function neuronadd(net, idx, vals)
    idx == 0 ? error("Theres an error in your config file: Neuron index can't be 0") : "" ## Actually can, but forbidding simplifies julia numbering
    if ismatch(r"^I[zZ]", vals["type"])
        params = vals["params"]
        addizneuron(net, idx, params["a"], params["b"], params["c"], params["d"], params["u"], params["v"], params["s"])
    else error("Only IZ neurons are supported now!") ## FIXME
    end

end

## TODO: move to spikesIO?
"Add a dict of neurons to net. Returns neurons added #"
function neuronsadd(net, nlist)
    for (idx, neuron) in enumerate(nlist)
        neuronadd(net, idx, neuron)
    end
    length(nlist)
end

"Private: add 1 synapse per dest to the net"#source, dest, delay, weight, plastic
function synapseadd(net, s)
    for dest in s["to"]
        addsynapse(net,
                   s["from"],
                   dest,
                   s["values"]["latency"],
                   s["values"]["intensity"],
                   s["values"]["learning"])
    end
end

## TODO: move to spikesIO?
"Add a dict of synapses (improted from JSON) to net"
synapsesadd(net, sdict) = [ synapseadd(net, s) for s in sdict ]

end ## end module

## TODO: move the following to spikesIO
## TODO: refactor synapse definiton
## "1": {
##       "from": 1,
##       "to": [2],
##       "latency": 4,
##       "strength": -20,
##       "learning": false
##     },
