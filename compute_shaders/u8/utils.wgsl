// the value is in little endian . 
// So bytes abcd is represented as dcba
//
const LEFT_EXTRACTOR: u32 = 0x000000ffu;
const MID_LEFT_EXTRACTOR: u32 = 0x0000ff00u;
const MID_RIGHT_EXTRACTOR: u32 = 0x00ff0000u;
const RIGHT_EXTRACTOR: u32 = 0xff000000u;

fn get_left_byte(data: u32) -> u32 {
    return  data & LEFT_EXTRACTOR;
}

fn get_mid_left_byte(data: u32) -> u32 {
    return  (data & MID_LEFT_EXTRACTOR) >> 8u;
}

fn get_mid_right_byte(data: u32) -> u32 {
    return  (data & MID_RIGHT_EXTRACTOR) >> 16u;
}

fn get_right_byte(data: u32) -> u32 {
    return  (data & RIGHT_EXTRACTOR) >> 24u;
}