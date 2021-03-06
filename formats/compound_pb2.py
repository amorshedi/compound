# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: compound.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='compound.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x0e\x63ompound.proto\"\'\n\x04Vec3\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\" \n\x04Vec2\x12\x0b\n\x03id1\x18\x01 \x01(\x03\x12\x0b\n\x03id2\x18\x02 \x01(\x03\"\x97\x01\n\x08\x43ompound\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x1b\n\x08\x63hildren\x18\x02 \x03(\x0b\x32\t.Compound\x12\x12\n\x03pos\x18\x03 \x01(\x0b\x32\x05.Vec3\x12\x1a\n\x0bperiodicity\x18\x04 \x01(\x0b\x32\x05.Vec3\x12\x0e\n\x06\x63harge\x18\x05 \x01(\x02\x12\n\n\x02id\x18\x06 \x01(\x03\x12\x14\n\x05\x62onds\x18\x07 \x03(\x0b\x32\x05.Vec2b\x06proto3'
)




_VEC3 = _descriptor.Descriptor(
  name='Vec3',
  full_name='Vec3',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='Vec3.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='Vec3.y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='z', full_name='Vec3.z', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=18,
  serialized_end=57,
)


_VEC2 = _descriptor.Descriptor(
  name='Vec2',
  full_name='Vec2',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id1', full_name='Vec2.id1', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='id2', full_name='Vec2.id2', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=59,
  serialized_end=91,
)


_COMPOUND = _descriptor.Descriptor(
  name='Compound',
  full_name='Compound',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='Compound.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='children', full_name='Compound.children', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pos', full_name='Compound.pos', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='periodicity', full_name='Compound.periodicity', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='charge', full_name='Compound.charge', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='id', full_name='Compound.id', index=5,
      number=6, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bonds', full_name='Compound.bonds', index=6,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=94,
  serialized_end=245,
)

_COMPOUND.fields_by_name['children'].message_type = _COMPOUND
_COMPOUND.fields_by_name['pos'].message_type = _VEC3
_COMPOUND.fields_by_name['periodicity'].message_type = _VEC3
_COMPOUND.fields_by_name['bonds'].message_type = _VEC2
DESCRIPTOR.message_types_by_name['Vec3'] = _VEC3
DESCRIPTOR.message_types_by_name['Vec2'] = _VEC2
DESCRIPTOR.message_types_by_name['Compound'] = _COMPOUND
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Vec3 = _reflection.GeneratedProtocolMessageType('Vec3', (_message.Message,), {
  'DESCRIPTOR' : _VEC3,
  '__module__' : 'compound_pb2'
  # @@protoc_insertion_point(class_scope:Vec3)
  })
_sym_db.RegisterMessage(Vec3)

Vec2 = _reflection.GeneratedProtocolMessageType('Vec2', (_message.Message,), {
  'DESCRIPTOR' : _VEC2,
  '__module__' : 'compound_pb2'
  # @@protoc_insertion_point(class_scope:Vec2)
  })
_sym_db.RegisterMessage(Vec2)

Compound = _reflection.GeneratedProtocolMessageType('Compound', (_message.Message,), {
  'DESCRIPTOR' : _COMPOUND,
  '__module__' : 'compound_pb2'
  # @@protoc_insertion_point(class_scope:Compound)
  })
_sym_db.RegisterMessage(Compound)


# @@protoc_insertion_point(module_scope)
