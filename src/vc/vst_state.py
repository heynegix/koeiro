from pathlib import Path
import struct


def set_model_preset(original, model_path, voice, pitch):
    """Replace documented ParameterState entries inside a standard VST preset.

    The official processor.cc wraps common/parameter_state.cc in a uint32
    length; entries are int16 ID, int32 variant, then int/double/UTF8 string.
    Every bound/type is validated before sending any state to the plugin.
    """
    if original[:4] != b'VST3' or len(original)<48:
        raise ValueError('Invalid VST preset header')
    offset = struct.unpack_from('<q',original,40)[0]
    if offset<48 or offset+8>len(original) or original[offset:offset+4]!=b'List':
        raise ValueError('Invalid VST chunk list')
    count = struct.unpack_from('<i',original,offset+4)[0]
    if not 1<=count<=16 or offset+8+20*count>len(original):
        raise ValueError('Invalid VST chunk count')
    chunks = []
    found=set()
    for i in range(count):
        tag, pos, length = struct.unpack_from('<4sqq',original,offset+8+20*i)
        if pos<48 or length<0 or pos+length>offset:
            raise ValueError('Invalid VST chunk extent')
        data = original[pos:pos+length]
        if tag == b'Comp':
            if len(data)<4 or struct.unpack_from('<I',data)[0]!=len(data)-4:
                raise ValueError('Invalid Beatrice component size')
            cursor, entries = 4, []
            while cursor<len(data):
                if cursor+6>len(data):
                    raise ValueError('Truncated Beatrice state')
                start = cursor
                identifier, variant = struct.unpack_from('<hi',data,cursor)
                cursor += 6
                if variant == 0:
                    cursor += 4
                elif variant == 1:
                    cursor += 8
                elif variant == 2:
                    if cursor+4>len(data):
                        raise ValueError('Truncated string')
                    size = struct.unpack_from('<i',data,cursor)[0]
                    if not 0<=size<=32768:
                        raise ValueError('Invalid string size')
                    cursor += 4+size
                else:
                    raise ValueError('Unknown state variant')
                if cursor>len(data):
                    raise ValueError('Truncated state value')
                entry = data[start:cursor]
                if identifier in (1,2,4):
                    if identifier in found or variant != {1:2,2:0,4:1}[identifier]:
                        raise ValueError('Unsupported Beatrice parameter schema')
                    found.add(identifier)
                if identifier == 1 and variant == 2:
                    text = str(Path(model_path).resolve()).encode('utf-8')
                    entry = struct.pack('<hii',1,2,len(text))+text
                elif identifier == 2 and variant == 0:
                    entry = struct.pack('<hii',2,0,voice)
                elif identifier == 4 and variant == 1:
                    entry = struct.pack('<hid',4,1,pitch)
                entries.append(entry)
            component = b''.join(entries)
            data = struct.pack('<I',len(component))+component
        chunks.append((tag,data))
    if found != {1,2,4}:
        raise ValueError('Required Beatrice model/voice/pitch parameters missing')
    content = bytearray(original[:48])
    table = []
    for tag,data in chunks:
        table.append(struct.pack('<4sqq',tag,len(content),len(data)))
        content.extend(data)
    struct.pack_into('<q',content,40,len(content))
    content.extend(b'List'+struct.pack('<i',len(table))+b''.join(table))
    return bytes(content)
