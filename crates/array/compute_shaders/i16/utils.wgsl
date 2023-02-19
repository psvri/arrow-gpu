// the value is in little endian . 
// So bytes abcd is represented as dcba
//
const MAX_I16: i32 = 0x0000ffff;
const LEFT_VALUE_EXTRACTOR: i32 = 0x0000ff7f;
const LEFT_EXTRACTOR_SIGN: i32 = 0x00000080;
//const RIGHT_VALUE_EXTRACTOR: u32 = 0xff7f0000;
const RIGHT_EXTRACTOR_SIGN: i32 = 0x00800000;


fn get_left_half(data: i32) -> i32 {
    let sign = data & LEFT_EXTRACTOR_SIGN;
    let value = (data & LEFT_VALUE_EXTRACTOR);
    if sign == 0 {
        return value;
    } else {
        return value | sign | (MAX_I16 << 16u);
    }
}

fn get_right_half(data: i32) -> i32 {

    let sign = (data & RIGHT_EXTRACTOR_SIGN) >> 16u;
    let value = ((data & (LEFT_VALUE_EXTRACTOR << 16u)) >> 16u);
    if sign == 0 {
        return value;
    } else {
        return value | sign | (MAX_I16 << 16u);
    }
}