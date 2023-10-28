// the value is in little endian . 
// So bytes abcd is represented as dcba
// 2 numbers 'ab' 'cd' would be represented as 'ba' 'dc'
// Hence in little endian i32 it becomes cdab

const MAX_I16: i32 = 0x0000ffff;
const SHIFTED_MAX_I16: i32 = MAX_I16 << 16u;
const LEFT_SIGN_EXTRACTOR: i32 = 0x00008000;
const RIGHT_SIGN_EXTRACTOR: i32 = -0x80000000;


fn get_left_half(data: i32) -> i32 {
    let sign = data & LEFT_SIGN_EXTRACTOR;
    if sign == 0 {
        return (data & MAX_I16);
    } else {
        return (data & MAX_I16) | (SHIFTED_MAX_I16);
    }
}

fn get_right_half(data: i32) -> i32 {
    let sign = (data & RIGHT_SIGN_EXTRACTOR) >> 16u;
    if sign == 0 {
        return (data & SHIFTED_MAX_I16) >> 16u;
    } else {
        return ((data & SHIFTED_MAX_I16) >> 16u) | SHIFTED_MAX_I16;
    }
}