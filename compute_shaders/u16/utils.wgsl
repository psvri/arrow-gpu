// the value is in little endian . 
// So bytes abcd is represented as dcba
//
const LEFT_EXTRACTOR: u32 = 0x0000ffffu;
const RIGHT_EXTRACTOR: u32 = 0xffff0000u;

fn get_left_half(data: u32) -> u32 {
    return  data & LEFT_EXTRACTOR;
}

fn get_right_half(data: u32) -> u32 {
    return  (data & RIGHT_EXTRACTOR) >> 16u;
}