#version 310 es
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0) buffer Input {
    uint values[256];
} sb_in;
layout (binding = 1) buffer Output {
    uint values[256];
} sb_out;
void main (void) {
    uvec3 size           = gl_NumWorkGroups * gl_WorkGroupSize;
    uint numValuesPerInv = uint(sb_in.values.length()) / (size.x*size.y*size.z);
    uint groupNdx        = size.x*size.y*gl_GlobalInvocationID.z + size.x*gl_GlobalInvocationID.y + gl_GlobalInvocationID.x;
    uint offset          = numValuesPerInv*groupNdx;

    for (uint ndx = 0u; ndx < numValuesPerInv; ndx++)
        sb_out.values[offset + ndx] = ~sb_in.values[offset + ndx];
}

