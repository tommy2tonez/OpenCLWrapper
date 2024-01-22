kernel void add( global const uchar * a,
                 global const ulong *b,
                 global ulong *c){
     size_t i = get_global_id(0);
     c[i] = b[i] + a[i] + 1;
}
