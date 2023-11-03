module KALaunchConfig

using KernelAbstractions

abstract type launchConfig end

function (cfg::launchConfig)(args...) end

function launch_config(kernelObj, args...; workgroupsize, ndrange)::launchConfig end

##
# Needed because of disconnection KernelAbstraction and the hardware size
##
function getStrideSize(::launchConfig) end

function getMaxGroups(::launchConfig) end

function getMaxWorkItems(::launchConfig) end

function setWorkItems(::launchConfig, workitems) end

end




