require 'nn'

local fwrite = function(tensor, file)
    if not tensor then return false end
    local n = tensor:nElement()
    local s = tensor:storage()
    -- return assert(file:writeDouble(s) == n)
    file:writeFloat(s)
end

local mdl = torch.load('model.net')
module1_weight = mdl.modules[1].weight
local module1_bias = mdl.modules[1].bias

local file = torch.DiskFile('modeloutput.bin', 'w'):binary()
fwrite(module1_weight, file)
fwrite(module1_bias, file)