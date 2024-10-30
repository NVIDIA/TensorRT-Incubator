module @jit_run attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x1000xf32> {
    %cst = stablehlo.constant dense_resource<__elided__> : tensor<1x1000xf32>
    %cst_0 = stablehlo.constant dense_resource<__elided__> : tensor<1024x1000xf32>
    %cst_1 = stablehlo.constant dense<4.900000e+01> : tensor<1x1024xf32>
    %cst_2 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_3 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_4 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_5 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1024xf32>
    %cst_6 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x4096xf32>
    %cst_7 = stablehlo.constant dense_resource<__elided__> : tensor<1024x4096xf32>
    %cst_8 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_9 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_10 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_11 = stablehlo.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_12 = stablehlo.constant dense_resource<__elided__> : tensor<169x32xf32>
    %cst_13 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x3072xf32>
    %cst_14 = stablehlo.constant dense_resource<__elided__> : tensor<1024x3072xf32>
    %cst_15 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_16 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_17 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_18 = stablehlo.constant dense_resource<__elided__> : tensor<4096x1024xf32>
    %cst_19 = stablehlo.constant dense<4.471500e-02> : tensor<1x49x4096xf32>
    %cst_20 = stablehlo.constant dense<0.797884583> : tensor<1x49x4096xf32>
    %cst_21 = stablehlo.constant dense<1.000000e+00> : tensor<1x49x4096xf32>
    %cst_22 = stablehlo.constant dense<5.000000e-01> : tensor<1x49x4096xf32>
    %cst_23 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x4096xf32>
    %cst_24 = stablehlo.constant dense_resource<__elided__> : tensor<1024x4096xf32>
    %cst_25 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_26 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_27 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_28 = stablehlo.constant dense_resource<__elided__> : tensor<1024x1024xf32>
    %cst_29 = stablehlo.constant dense_resource<__elided__> : tensor<169x32xf32>
    %cst_30 = stablehlo.constant dense<0.176776692> : tensor<1x32x49x32xf32>
    %cst_31 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x3072xf32>
    %cst_32 = stablehlo.constant dense_resource<__elided__> : tensor<1024x3072xf32>
    %cst_33 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_34 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_35 = stablehlo.constant dense<1.024000e+03> : tensor<1x49xf32>
    %cst_36 = stablehlo.constant dense_resource<__elided__> : tensor<2048x1024xf32>
    %cst_37 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_38 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_39 = stablehlo.constant dense<9.99999997E-7> : tensor<1x49x1xf32>
    %cst_40 = stablehlo.constant dense<0.000000e+00> : tensor<1x49xf32>
    %cst_41 = stablehlo.constant dense<2.048000e+03> : tensor<1x49xf32>
    %c = stablehlo.constant dense<1> : tensor<7xi32>
    %c_42 = stablehlo.constant dense<2> : tensor<7xi32>
    %c_43 = stablehlo.constant dense<0> : tensor<7xi32>
    %cst_44 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_45 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_46 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_47 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_48 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_49 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_50 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_51 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_52 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_53 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_54 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_55 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_56 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_57 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_58 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_59 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_60 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_61 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_62 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_63 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_64 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_65 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_66 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_67 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_68 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_69 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_70 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_71 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_72 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_73 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_74 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_75 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_76 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_77 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_78 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_79 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_80 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_81 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_82 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_83 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_84 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_85 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_86 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_87 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_88 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_89 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_90 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_91 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_92 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_93 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_94 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_95 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_96 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_97 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_98 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_99 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_100 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_101 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_102 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_103 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_104 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_105 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_106 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_107 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_108 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_109 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_110 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_111 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_112 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_113 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_114 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_115 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_116 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_117 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_118 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_119 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_120 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_121 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_122 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_123 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_124 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_125 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_126 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_127 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_128 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_129 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_130 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_131 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_132 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_133 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_134 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_135 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_136 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_137 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_138 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_139 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_140 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_141 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_142 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_143 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_144 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_145 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_146 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_147 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_148 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_149 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_150 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_151 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_152 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_153 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_154 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_155 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_156 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_157 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_158 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_159 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_160 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_161 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_162 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_163 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_164 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_165 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_166 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_167 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_168 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_169 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_170 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_171 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_172 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_173 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_174 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_175 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_176 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_177 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_178 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_179 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_180 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_181 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_182 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_183 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_184 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_185 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_186 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_187 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_188 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_189 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_190 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_191 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_192 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_193 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_194 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_195 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_196 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_197 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_198 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_199 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_200 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_201 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_202 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_203 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_204 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_205 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_206 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_207 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_208 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_209 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_210 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_211 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_212 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_213 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_214 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_215 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_216 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_217 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_218 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_219 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_220 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_221 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_222 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_223 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_224 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_225 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_226 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_227 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_228 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_229 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_230 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_231 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_232 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_233 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_234 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_235 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_236 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_237 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_238 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_239 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_240 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_241 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_242 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_243 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_244 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_245 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_246 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_247 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_248 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_249 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_250 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_251 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_252 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_253 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_254 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_255 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_256 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_257 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_258 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_259 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_260 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_261 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_262 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_263 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_264 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_265 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_266 = stablehlo.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_267 = stablehlo.constant dense<4.471500e-02> : tensor<1x196x2048xf32>
    %cst_268 = stablehlo.constant dense<0.797884583> : tensor<1x196x2048xf32>
    %cst_269 = stablehlo.constant dense<1.000000e+00> : tensor<1x196x2048xf32>
    %cst_270 = stablehlo.constant dense<5.000000e-01> : tensor<1x196x2048xf32>
    %cst_271 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x2048xf32>
    %cst_272 = stablehlo.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %cst_273 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_274 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_275 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_276 = stablehlo.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_277 = stablehlo.constant dense_resource<__elided__> : tensor<169x16xf32>
    %cst_278 = stablehlo.constant dense<0.176776692> : tensor<4x16x49x32xf32>
    %cst_279 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1536xf32>
    %cst_280 = stablehlo.constant dense_resource<__elided__> : tensor<512x1536xf32>
    %cst_281 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_282 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_283 = stablehlo.constant dense<5.120000e+02> : tensor<1x196xf32>
    %cst_284 = stablehlo.constant dense_resource<__elided__> : tensor<1024x512xf32>
    %cst_285 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_286 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_287 = stablehlo.constant dense<9.99999997E-7> : tensor<1x196x1xf32>
    %cst_288 = stablehlo.constant dense<0.000000e+00> : tensor<1x196xf32>
    %cst_289 = stablehlo.constant dense<1.024000e+03> : tensor<1x196xf32>
    %c_290 = stablehlo.constant dense<1> : tensor<14xi32>
    %c_291 = stablehlo.constant dense<2> : tensor<14xi32>
    %c_292 = stablehlo.constant dense<0> : tensor<14xi32>
    %cst_293 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256xf32>
    %cst_294 = stablehlo.constant dense_resource<__elided__> : tensor<1024x256xf32>
    %cst_295 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_296 = stablehlo.constant dense_resource<__elided__> : tensor<256x1024xf32>
    %cst_297 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256xf32>
    %cst_298 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256xf32>
    %cst_299 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256xf32>
    %cst_300 = stablehlo.constant dense_resource<__elided__> : tensor<256x256xf32>
    %cst_301 = stablehlo.constant dense_resource<__elided__> : tensor<169x8xf32>
    %cst_302 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x768xf32>
    %cst_303 = stablehlo.constant dense_resource<__elided__> : tensor<256x768xf32>
    %cst_304 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256xf32>
    %cst_305 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256xf32>
    %cst_306 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256xf32>
    %cst_307 = stablehlo.constant dense_resource<__elided__> : tensor<1024x256xf32>
    %cst_308 = stablehlo.constant dense<4.471500e-02> : tensor<1x784x1024xf32>
    %cst_309 = stablehlo.constant dense<0.797884583> : tensor<1x784x1024xf32>
    %cst_310 = stablehlo.constant dense<1.000000e+00> : tensor<1x784x1024xf32>
    %cst_311 = stablehlo.constant dense<5.000000e-01> : tensor<1x784x1024xf32>
    %cst_312 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x1024xf32>
    %cst_313 = stablehlo.constant dense_resource<__elided__> : tensor<256x1024xf32>
    %cst_314 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256xf32>
    %cst_315 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256xf32>
    %cst_316 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256xf32>
    %cst_317 = stablehlo.constant dense_resource<__elided__> : tensor<256x256xf32>
    %cst_318 = stablehlo.constant dense_resource<__elided__> : tensor<169x8xf32>
    %cst_319 = stablehlo.constant dense<0.176776692> : tensor<16x8x49x32xf32>
    %cst_320 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x768xf32>
    %cst_321 = stablehlo.constant dense_resource<__elided__> : tensor<256x768xf32>
    %cst_322 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256xf32>
    %cst_323 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x256xf32>
    %cst_324 = stablehlo.constant dense<2.560000e+02> : tensor<1x784xf32>
    %cst_325 = stablehlo.constant dense_resource<__elided__> : tensor<512x256xf32>
    %cst_326 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_327 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_328 = stablehlo.constant dense<9.99999997E-7> : tensor<1x784x1xf32>
    %cst_329 = stablehlo.constant dense<0.000000e+00> : tensor<1x784xf32>
    %cst_330 = stablehlo.constant dense<5.120000e+02> : tensor<1x784xf32>
    %c_331 = stablehlo.constant dense<1> : tensor<28xi32>
    %c_332 = stablehlo.constant dense<2> : tensor<28xi32>
    %c_333 = stablehlo.constant dense<0> : tensor<28xi32>
    %cst_334 = stablehlo.constant dense<"0x3251893E5F40A7BD1D545CBE7E318B3EDD75C0BD226FC63DDD3CA13D446586BE4631013E05A41E3CCA8F6B3E6BA380BE4D3D933B39602ABD028B1D3DA5044FBEDB0AA03D0B9345BC9B8C35BC838CB3BD56F2A93CEC12903D9043883E871890BECC19E23D9CDF0FBEECC7E23C15E8703C97D6D23D7A2D84BD0D0CB13BCEC92FBDBD60433EA91731BDDD0B2B3E6EF7CDBDE87AF33E6BF0C03D6819173E8082933EF616F73E3E77133E0B132C3E334A923C82888ABEDF43383ED4D3C1BD55F94D3DA16EF8BE9759003E4136E13E6EDB2F3E994A99BED74C3F3D5D5A01BDD1AB153E50CC8B3F9CD1C8BD9EBBC43DCF66BC3E3B81403EAEC9A03C9D3C103D85BEBDBD301D03BDFF2AF6BD4546483D8EE20D3EA437E9BD0F4A4A3DAEE5253D1D4D5DBD1C238ABE2CD8F63E7ED2DDBE300B89BE650F923E71C2BDBEA1EB1EBE873305BF8B9698BD849FC23D0C47203E2D57C8BFFBBB933D4BE3E3BE190567BE0E7B263E82FF0FBD781F043D8589533DE1C59CBDB52742BD855E3ABE2E8D39BE210766BEDF51A8BE9BE190BD97FB70BD919DE03D1CE1B33C0EA7E3BCE1D21BBD3072F0BDCBFD25BE2E6BE0BDE0A388BE50E3553E33669D3C050D9A3EF750D73E2DC52ABEFDD680BE6DCE883EBA998DBDDF6047BBED907B3D2F44C3BD022CD03DAB21D6BD189C493EA0303A3E1728C1BDEB69B63FBACB153EEFEBA2BEA599C9BD6015753E"> : tensor<1x1x128xf32>
    %cst_335 = stablehlo.constant dense_resource<__elided__> : tensor<512x128xf32>
    %cst_336 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_337 = stablehlo.constant dense_resource<__elided__> : tensor<128x512xf32>
    %cst_338 = stablehlo.constant dense<"0xB432903E34E6C3BDBB19E6BEC136EB3EF0A9823EA077B6BD42EC66BDF71610BE4F865E3F53795FBEA39A9A3B6A70D73C6725263D240F65BEDF0AAD3C12CB8A3B79AA6A3EE2190A3ECCA5923E1C239F3EADE0BEBE00EDFC3D9B2127BCF41C983D02CC0B3EEAFB77BE32D04DBC8CCF08BE0AC4743E608505BED4BC063F2EBE87BCC617183E8885AC3ED881C93EEDE391BE4DC0C23CBD292A3EAB2C213E5D320CBE29C33A3E69969ABA7C16043DB35EC4BE7B7761BE1398213F0D9866BE8E106FBE4C471CBFC538B9BEC15EE43DD69A0ABECD1A42BD217C073D08805ABE1FEBD13DE41B20BE9ABDD03D918340BEE00E1C3E8F1D5DBD38CB663DCBD994BB29F116BD7C86463E9EC557BE45FEEABD30CA533EF799893E4BD081BD4EE6333E51A5453E050ADA3D66E9923CB050EABE2D17BEBE924CFCBDD6415C3E25B780BF9C078EBEF2E7F8BD1A36A63E4EFC863EF239143FF6206ABC80FAF73E98E1D9BEC4B91D3E1E2D703D1317AFBD79746C3E68C72EBE1D2D3A3B9161293E1DD5B03DA936983E66AE1DBE95AC48BEF0C9E73DC34951BC67D54DBE94D1C53E781F0D3EA454D1BE44E1B43E33B529BCC99B93BD0F35473E1B4409BD005210BE03B7A73EDBBF153E86A06E3D541C403F80621CBE5E0222BEB3B364BDEA20533E10BA433C559DEEBDDCDC623EFC767BBE9E17B9BD004D37BB9AA8603EDB2683BE7D4DCFBCCB7A473E"> : tensor<1x1x128xf32>
    %cst_339 = stablehlo.constant dense<"0x9E3B0A405937F43FA515FC3FB104DC3FE9C0FE3F6DEC0C405EB80B40CE43BA3F6E74C33F2F43863FB882CA3F7F75E43F2405EB3FDDB3D63FABF4EC3FB06F0440449F1040BC260840E83DE23F35BF0340D82E18407D4DA13F55190840C00A973F022CEF3F2E9ACF3F432B064050E00440BF60C63F7A0BF83FA113F03F9C73D83F9F181440C16CFD3F1B170740434AF93FA5F90440F59214409161393FADA709407403FC3FE65AF23F6CEFED3FA060E93FFE6DF33F39F5E03FC741014050E00B4098E2EF3FE540E23F16EDFC3F24F00140BDA5F43F7A3AF03F36A4D73FE3DA0E40AE8C623F7BDFAB3F9FAAFA3F24E1A23FA1C7F13FFF8C0240E40D044097B702400E08FF3F6425E93F37A00F40A23302409499C33FF4810640761B0A40A3E60F40860C6F3F4B6EF93F9E3DEA3F636FC83F457CDF3FD55B6A3FB15DB93F240FFC3F6A34DF3F4C50EB3F3579FB3FAD6B923EF8D60040AD6BB43F4FF9F23F32B9EA3FB588D53FBAFFEC3F8E121240633C10406E7C114008D4983FCA59CE3F9673AA3F3F0AF13FD89CDC3F06B1FB3F2F51E33FDFE8F93F4CC7024097F5E43FCAC0A83FFF77ED3FD2F88E3F1AB8EF3F2EB0EB3F7BB6E83F1511F33F36E3BE3F61D8DA3FAC2D084092A504408425FA3F9CD3DF3FBD9A8C3FF1E7DE3F401AF43F7829CF3F1123F83FD279F43F6FE90D40F8277F3ACC251440944AE73F7C230440779AFA3F"> : tensor<1x1x128xf32>
    %cst_340 = stablehlo.constant dense<"0xBAD74B3DBB6528BE332CCA3D2496B5BEAA2A983D99FABF3E7B0BA23E846276BD8102123EEF2FE43E076283BDB85644BE29B3803ED483BA3E7241683C857A8FBEB0E921BEA9A8DEBDE2340FBE2D4C233ED93A5FBE00E5893BE2572D3DE5D8373D4114CEBE1285A13EEFCE6CBE53C59DBEB559E2BD5EF9583E4385953D8729193E18E0993EE95ABCBEF89255BD45E35B3E5CADCF3E548BC9BE079A09BDA469AEBEA5BCFFBD88088D3E6A612ABCD615723E2847373E7EBD24BE3E00E13D60895E3E26EF6F3D3178573E1BADFDBD451585BE3AF27F3D155A4E3E44E941BD5FA399BDBAFFF4BEBDBAD43E1D07E43E039566BEC3AF01BE5BBD143E4ACF003EB30DDD3DA4AD493BE5D0583E992D7A3E1DEBECBDD87690BEDE68F73DC691C0BDABB8813E1BE1AC3EF14A1EBE4006BD3E3C42A43DF41521BE002B7CBE836D9DBE536B903EF10C25BE2B6C16BE0530953EAF24EF3D2F7EAEBE06C587BE3984C93E8F6D45BE0C37983E3513D2BED8B11B3D41618BBE7B5F9D3DC0191F3C38222D3EA90C2D3E977EC7BD194D933EA131BEBEC9A2A0BC38AA9E3D105586BE73E633BF64F0B93E54C3AABD0F3A28BED9C5493ED0A8ACBD4B1A16BFC441C63D8E04213CD797733E4A3A5ABEE74C29BEC515C43D111DA83DB02631BE0B2957BDDC5CDB3E60C8C3BE0579983DF2FDD63CC1EE8ABC4F4585BCDF43713C9AEA96BCF4A51F3E8122FFBD"> : tensor<1x1x128xf32>
    %cst_341 = stablehlo.constant dense_resource<__elided__> : tensor<128x128xf32>
    %cst_342 = stablehlo.constant dense_resource<__elided__> : tensor<169x4xf32>
    %cst_343 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x384xf32>
    %cst_344 = stablehlo.constant dense_resource<__elided__> : tensor<128x384xf32>
    %cst_345 = stablehlo.constant dense<"0xD1621E3EC3897CBE580942BE8F2582BEBF19313EAF3E993E9A47AD3E336BDABDC3CA123F4A9FA0BDB993A5BD766BCABD3D3B583E2D4C6A3E694EA33C6AE19ABE348100BEDCF43BBD0933683D0028D43EE959ABBEC6B352BE2414F83B8DDDE0BD098E52BEF3926B3E57CD74BEC3AFC6BEA9E3CA3DC9E0623E45369E3E260AA83E9D4A973EF7AE3FBECAB0CF3D47A0AF3DABCBF53E9D1EFDBECCD5833E64B7C2BE12FDF2BB25E27D3E7346483DE750C0BCC62D043ECBF26BBE788FCA3C5A956F3ED64F80BEE1BEC7BB491C19BE8516BABE872F313DA5C7433EEC38BFBD0FEE8EBCEB4801BFDD23C23EB68D973E92C618BE66BA9ABE03862D3E105A2D3EFA13003E6896C6BDB3401A3E4B29783E8049983C7982C0BC57AE2A3D69D2EBBC975FAB3E005A9F3FB187E9BD09C9183EDE5CDEBD10FD8FBE85F146BFEFAB423E9547533EC19659BED738723DDF779C3E03C3E13F3EF6CBBE7096963D67E42B3EFCFAD8BEA09EB83E4077E6BE357FCB3DADFFCEBEF25975BCCD9D42BE6B090C3EF200A93D378B69BEF5B1273EA7A9AABEAB6E083DFACD71BD742ECABC3B97FBBE1B56233EBFB2673CA85B8DBE2ECD0F3ED6E8B3BDFB1CE0BE9AEB87BD6B77693EC272893E10DE4EBD5A895BBD94C4113E873EB9BC1882A9BE88F06D3EE93CE13E90AAADBEB9BB413E3B85F5BD69030EBE41D98DBF4AC082BD16049FBE4DDF483E0A92F33C"> : tensor<1x1x128xf32>
    %cst_346 = stablehlo.constant dense<"0x7FBD853FF5B98B3F18D0963F15626D3F7586933F32728E3FAEE7A03FB7135D3F22DA6F3F47A2253F0B8B613F01E0793F5753913FDE1E803F7E71803F348C753F467CAB3FFE279C3FC4B0893FDF8F993F4822A43F5AB5443FCAA6A83F4C19733FC512943F7561523FB0428A3FD1D2953F4C324D3F7D3C803F0CEC8E3F4CF26B3F64D3983F3710883FCDFC903FD881793F81C2C33F39969A3FD4F0EC3EDE2E893FECFB8D3F13DF893F6FFD873F2440933FF1B49A3FBBB2633F0B1C593FD527863FBBE07D3F27A1A63F87C7883F9F5C8B3F33DB973F4832493F8746703F06DDBB3FD3570C3FF8F7173F1C278D3F6D2D183F8295563F3E888B3F39D1A03F02D9A73F2C6E863F22C6823F0C7B983F7AA18F3F39F33B3FFF99953F7305A43FC24F703FD8855C3F57DA8C3FE193923F2B13613F7B00673F5584373F63DD323F0BE6923F10016B3F5182923F117B8C3F5A68C63EE7FC9F3FAEE7873F8751853F1D68883FB37D4C3F61548A3F0CDC973F1789943FE3B69F3FBE040C3F6D068E3FA755383F807C813F2B7F853FA54B843F2A0C8D3F59D0923F89E3AD3FAED1563F593E343F304CCE3FC459203F1250843F59D19A3F467E6C3F849C8D3F5C878B3FD281773FEAC99A3F97F25F3FBF66A13F18C5873FC358773F17E76F3F99E06D3F409F503F37ABB43F2DEC973F60519F3F6EE4813E3606C13FB60D8A3F85328C3F2177993F"> : tensor<1x1x128xf32>
    %cst_347 = stablehlo.constant dense<"0xC1420D3D6913E73D79C0ABBDA160873E7262F83C415C70BEFDEAA23D1226F8BB3921C4BC438FB53DCBA5B63EB58AC5BD8305FA3D41E14EBE6E4532BE62CCB73B4B4939BCE7A22CBE95889C3E9889203E4AB636BDF8A45CBDBBB92D3D0453A63E2CB7D9BD1D8A8ABE8407A4BD6F6F563E5FA35D3D496FA4BD43F138BE38C8DDBE4C16923DFDD2283D6579BDBDCE258B3E35EF173EFF5DD93EBD4485BEA3324DBE1B159A3EB0F612BE927181BE2F9659BDCE67C23D5759613E09B5FABD90E9823D4693A7BD64630E3DF37316BE5F59AA3D29E555BD89E883BED9F5ECBC9C3E923E17A9613FD54BAEBE9EAB0C3EDAAB9C3E42222C3D5EC9423EEB38EEBEBE91F33D7B70743D4A365ABC8E321CBD2F255F3EE3B6053E9B928C3E6523383E08FF98BD4C978BBF3214D53E9C99F63960DA563D6108473E820A943F9FCA85BE4B4F52BE07CDEE3DC358783E4EEF03BEBAFD28C0ADA30A3D162C31BEB5FF41BE0A2AC23EBD5C11BEE3C6CDBDE2E65DBDE3DC853BBC0E3D3EBB0D593E3A3DAABDF60DDCBDA96D533E690F8B3C51341B3E045972BC72F2903D86AE15BEC73E723E802908BEA1BDF83D1C47913E32244F3E0016283E57DB983D839E17BDCFB520BE77C4A7BEC40C303D465CDE3EBA222E3D1F6248BE22FD193F1FE7B5BD1C49EEBD7B4C28BEEE6A2FBE74375EBE0B9D423ED368BE3FD02A12BD5A5306BE190EA53DDDA5E5BE"> : tensor<1x1x128xf32>
    %cst_348 = stablehlo.constant dense_resource<__elided__> : tensor<512x128xf32>
    %cst_349 = stablehlo.constant dense<4.471500e-02> : tensor<1x3136x512xf32>
    %cst_350 = stablehlo.constant dense<0.797884583> : tensor<1x3136x512xf32>
    %cst_351 = stablehlo.constant dense<1.000000e+00> : tensor<1x3136x512xf32>
    %cst_352 = stablehlo.constant dense<5.000000e-01> : tensor<1x3136x512xf32>
    %cst_353 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x512xf32>
    %cst_354 = stablehlo.constant dense_resource<__elided__> : tensor<128x512xf32>
    %cst_355 = stablehlo.constant dense<"0xB7130A3E1FB274BFA3B998BEA02412BF919338BE541B5E3E587A0E3F8DD26FBE323E733FAE0AB13EBEFFA7BE670E72BEBB69ABBE4A51AD3ECF26BF3DBA7750BFE980C9BD015271BEA296CE3DCEF4073F8CBD0FBFEE1F1F3F614A903E0ECAD9BC8AB39ABEA5892E3F2B0C553D640379BCF2B137BDF0619E3D0489683E3986143F5C223FBEC75CECBD7FC4683E5F7024BE9D1ABE3E22F3A4BF800B303E84C997BE39FAC33E3A72503E20B78BBD622DA03ED3B12EBE0313F43D049936BD34B4543D47B694BECB846CBE93D80FBED917C8BEA513A4BCC3F0063F1F2FC6BEDE61A3BE8B73EDBD37AB1EBEA1A5A23E50381BBF3728C0BE91C96CBEF3FB36BDF3ABEC3DCA761CBE5B5BC2BE5837B63E8EF487BF93CDB53D6610893E23D128BE53C3933F353EC33EAEFE4B3E7A0B95BD84EE08BF5D6F0BBDD8B43CBF7FCD3EBF36CDE2BED0200F3DC773CF3DD93B873EAE31523F1AE5293D57F3B93ECE23A3BB34E12DBF06159B3E6508CCBE00BE8A3E496F5ABB817FDABDE289C9BE21FA1E3FD1570B3F8D250ABF2FE384BE36B5F9BE9E524E3E506759BE846AA9BC104D97BE3BD001BF303F163FCEDB96BEF88225BF2117F5BD05E653BEFD505FBD71DC073EF289763E1576EA3DB4A7143F7DB2863E2C7C98BE250F22BF2B0EC63EEC3DEC3EA3642CBEEEFD1F3F445E803C1C26DEBE05A207BFEB2764BF2A94A5BECF81A1BE7802803E"> : tensor<1x1x128xf32>
    %cst_356 = stablehlo.constant dense<"0xB34EA63F8363553F2C9F20405637BE3F2ACF943FF82DBC3F437B903FA7A6493F920C483F29F10F407191323FF84EBF3F0FDAB83FFBC66D3FC7D40740D1CD863F347B813F096FC53FA0F00C407443BE3F74FA9C3F54D0AC3F4D8F8E3F1BB6103FF37AA93F3244AF3FC77BA93F4ED103409C5C39405EFB5D3F9CCD9C3F944FD53F7B18623F7179D03F1560E63FF989FF3F6FC3943FBB9E8C3FBB8B773FD715F83F7C8DB93F1F250B40FA0C8E3F04250C407EEC1C4077260140331DF33F00D1393FB825F93F377F8C3F0F554D3F93D2873F95EB673F1112E03FF09D154093FF4C40F3B9D93EDF151B40AEFF263F6339F63F6E8F124003D06B3F43250E40E205963F0B88B23F48F0E93FB50F7C3F89EA853F746C3E3F3040953F0AD8F63FA984D53F3EB6B93F4310DF3FD0BA0940A466A43FB81E074041E8093F0983124082EEEC3F6A6430404B231340C6C74D3F3BA1D53F467BCF3F3FA1BE3F48A406402F9D943FBBE67F3F13930F40762B783F0F91373F31498E3F059B603F2AADA73FAFD4A03F21F6BE3F0B1B0940AF7F413FF1B7AF3F6F16D03F2B8C723F98E2F83F4D210840200C9B3F957F673F4891DA3F43E12140D19B673F727D0040F0D3833F923D104048667E3F6F43BA3F756810402C3BAE3F10075D3FE2965E3F620E104041E013408315603FE2099C3F7322AF3F2FC9B63ECE61D23FFB19603F04F202409AAF0A40"> : tensor<1x1x128xf32>
    %cst_357 = stablehlo.constant dense<"0xFEDC793DA6A44BBE641D19BE931A1DBFD3528A3D870F133E71BA0E3E3DD2D93DE3D46A3E4DD4133DA381CDBEDC5086BDAAB88FBD8C391CBD93EEFBBD1DD545BF5B5474BE306791BEB27C293EBB47F23E9986FFBE90D1B43E459202BE44FEB33C58F5C9BE0C99F73E5DCDB2BDB3E5C8BA7A2F273E513F123EEC75BC3C7AE7843E24C1793DDE97BABC6396A13D4B1462BD9E1F40BDF7C2B4BEA04AC4BC112B95BE7484C13E2713833D40C0C6BB210C2A3E7070D53DE265073E6FAB163DB75D953D36F239BED3E26EBDA86D293C004789BE3FF6893D3A0C9D3E385F5ABDA072B13BF7E24EBE92AD77BD43437D3E396107BF5E13F8BD12C04EBC93084CBDF2C7863E662605BE3E479CBD2442273F56CC89BE1487FB3EE7D6023F2432A13D2BC2813E3CBBAC3E7188B63D293B4F3D6A869CBEE08F18BEC8885ABFB38502BB120EAEBDFE1ABD3BB0B3EE3D17C7863E04B5DE3ED7EF82BDC8D9CB3D1B75143E008AB2BE584AA2BCC33021BE3573BF3DCEF3DDBEE3D78DBD7E03CCBE88DD0F3FFDE8F93DFEFAC0BD3752303EC8FDC0BE16FF68BD95403DBE19BE2A3D494707BEA29CC3BD33B6583EF8205EBE8F3846BE80F3463B00B0B63D31AC0FBDA907563DD852C93DFD49603EC17A233FD10EBE3D0DAA9FBE4E42F7BE9E309E3EF8511F3E45440DBED5D4933EC7088DBD1C2CBCBE8981C5BE301DB8BE466C08BF99DFB2BC42A200BE"> : tensor<1x1x128xf32>
    %cst_358 = stablehlo.constant dense_resource<__elided__> : tensor<128x128xf32>
    %cst_359 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c_360 = stablehlo.constant dense<169> : tensor<2401xi32>
    %c_361 = stablehlo.constant dense<0> : tensor<2401xi32>
    %c_362 = stablehlo.constant dense_resource<__elided__> : tensor<2401xi32>
    %cst_363 = stablehlo.constant dense_resource<__elided__> : tensor<169x4xf32>
    %cst_364 = stablehlo.constant dense<0.176776692> : tensor<64x4x49x32xf32>
    %cst_365 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x384xf32>
    %cst_366 = stablehlo.constant dense_resource<__elided__> : tensor<128x384xf32>
    %cst_367 = stablehlo.constant dense<"0x44B0EE3D92AE1ABFE5138FBE8F2A26BF5903EEB8C6F2A53E21B03B3EB95265397D81583FB1F4493EFED495BEA23E2BBE925185BE34185839A99E02BEE30B3CBF0B6619BEC89EC4BE478CAB3EA40A283F6FAFEEBEAB33EE3EBD7F0B3CD2E3D0B8CAA2F8BE81BFC53ED8944339063C3CBC20E34C3EA824013EDAB5093A6FE9903E62C069B842A8FCBD371A2B3E2C94CFBDF6F947B9F98D5BBFCEFF90B9664095BEB06DF33E68F1633E84D601B8FBD5CB3E49011E3ECADB8A3E1B878C3D8FBDD4B7BA49A4BE4904B8B999ACBF396C2F98BEBE6208B9422FAB3EC00705BE1F2863B7115CADBA07EDD1BD00DD2F3EE0C308BFBD4D53BE1C3C99BDE414A5BDE9E9723E26C328BE25450CBE6E66EE3EA6742FBFFDDF843E6D17ED3EAFAD013D14D1433FBA712B3F4A15533EB835C23D4772C8BE920A42BE897CBDBE1CB3F53D6CC577BE58ACA93D8411833EBCCE5B3ED52E523F2DCDDD38753E823EAF09803E7D0305BF4CCDCF395D1AABBE16D8473E87B72EBE8F82A4B9555086BEE5ACD13EED895B3EF61F54BE5B00883EF4808BBE46CCEB393EEF6CBE87F4093D9ACE9EBEF95780BE67EDDF3E83F756BEF707E6BE0108C03CC7C943B95A2F2A3D2772003907F1803E9054433E1A26453FE26C8B3E28A7D0BEB4F07DB9E6BE9B3E117DC63E9F4068BEDB93203F23A437B96C9DADBE55E763BEB0292DBF60B8D8BE5316A7BDBF679ABD"> : tensor<1x1x128xf32>
    %cst_368 = stablehlo.constant dense<"0x65EF3C3F94E9173FC331873F4EF60C3F6203433A0BEB413FB5DCA73E7686043A76434E3F4EC98B3FE7C69A3E7417543FE4546E3F91527CB9861B753F1022C03E404CD73E7F62263F95C9683FDD3A163FC846DC3EDBEF023F06E9CA3EDF3B12B96448183F0D7D043F3E573B38BEDB7D3FB672893FD250E33ED1786B3932482D3F93AE0138FFEC693F9C625C3FE01C6D3FB276AAB99F84433F4F7CF2382CB82A3F976B083FE6177D3FE2BF16BAF961763F048C8B3FEAAC633F4FE8623FC904363A981D683F1D412C3A65F7153A3326D63E5BADD839E101363FC388803F2F9E9D3909F80D3BFE048B3FEBF1383E7CFB483FF727843F55401D3FD6EA783F714CDE3E8F74593F93ED6F3FCF249C3EF234303FE603843EDCEADD3E8455353F6DFE643FF50ADA3E08C3583FD599823FD179FD3EA82F6A3FC247583EBA02943FB2C8773FD523843F1F34833F8340903E53B7903EA605ECB955203D3FA416713F7A5E183F1BB6C439143C6D3FBB6CED3EB82A813E0113E3B7A3499E3ED0B8DE3EEA94123F02FB3B3F3BF66F3FBC0E823E119C90B968A5413F7EC4963E1C03703F87D2843F1F9C1E3FD36FDC3ED78B473FEEF48A3FE4F778B9CD07813F150847396E157C3FA777DA3EA60B093F4EDA743F8D67123F77C1D5392922B63E33A4663F867C773F76FF2D3F75887139A433013FA1B00E3E192F5C3F091EA83E833C7B3F12587C3F"> : tensor<1x1x128xf32>
    %cst_369 = stablehlo.constant dense<"0xB1781A3EF2830B3F0792A4BE659287BFECDD0BBEB6E6AE3E81496FBEE78840BDA4223DBF964F293B70AECBBEDE0E27BE96450FBFA0DE28BEB07EA4BE1D5AA0BF4A962BBE7BFEEDBEFD2C123FAB698E3F6F5032BF2ADD033FFD5893BEBBC3F5BD5E905ABF458F103FE10B65BD2B530DBEBC16673DAFD6A13E8BCD19BE68D7B73E131C3CBDF4B8C8BE2A478F3E87B225BE40EC00BECE2D233F55470ABE438BD3BE679D4A3FC4297B3E005E4DBDCA85193F67364F3E668AEF3E5D798E3D947101BE6DFDD8BEB15DA73DB237C3BB9ACF71BEE69612BEF4B2B33E6820893C19D758BD04D5273E497A40BE668AF4BDF2571FBF6ECE7DBEBBFF06BDD242A7BE3815CCBEA67E55BEF4D2DFBD8518703F9E510B3F41C1053FFB69693F99DAC53DCE3220BFFE48CF3E7EA0C03ECEEDFCBD08EC7EBE2FE3ABBE443309BF68ECCDBCB2C98DBEB2B153BE86FAC83E809E1D3E7499833EE18A57BD3E67CF3D14F8E43D53CDE83EFCB924BE6424BABEE33BFC3EF1F201BF9E19C43CFDA08ABE38330B3F10BB303D9AE2533E286AED3E7415963D7604BBBD3EF01CBF4637033D097FDFBED35C87BEA80101BFBF6E39BE039FDABE7A881B3D46E7B13CE766193E07F0A0BDE89E2D3E5664893EE2DBCC3F6B15783E81573ABF47EFE5BEC9DD8D3EFD87A03EF3F8AFBE030725BFED67CEBCBE59D8BE35C108BEA2FAE03E64B738BFA4D435BE1D669DBE"> : tensor<1x1x128xf32>
    %cst_370 = stablehlo.constant dense<"0xCE8D1F40087093BC36352440D6893C3F425F4D3CD61D2B40545A803E1FAD663EDDC7E7BAA80165405FCF653F089C424047502E40946D683C583766406415A43FE4E4713FBCD7803F202D3940A684AE3F71C1713F2E8BB23FB0B4233FB055143C9F65873F616C9F3F2C05E53C7496684003BA133F6D20A03F84274F3D4E15C63FC3D17C3D4C764940A65548402D685C40F8EC5F3C0302B1BBEAEA943D641E0A3F0AA1A13FB1064F405BD9233DA3D33F40316334406D28B43FE5A90C4002C1503E81BC07404DEE363DF079133E1B90773F1CCBF53C069AF73F88205040CFB2CCBA0563DA3D906A5740AB10B53E0AE6533FBAC918404884D73FD46CF63F64565E3C8DC7374096FD5E40F248A93FD304313CF40C5F3F2111A33FC875403F61BC073B1E38123FAF9A3240B85D15405298163F0A6A4040E9941C3F269B8F3E5AC13940026F2B3F134054407864003FE495EA3E16FA933C2CD1DF3FC42415404624D2BA9E12373DED9E2540D8B7E53F9087923FB172BB3C790CEA3ED247623FE08CB73FB7748F3E8EDD19402C679F3E4B9BB73C0CBF42401845333F9D5A4140E27E64406C700F3CA9A37F3FDC1814406BB240401792003C094F5A405258173C04665D40763F343F44CE733FA8EE5140064AB93F8594193FD3C1273F99E30E40F0971A40754C3C3B683050B8AA9D283FFCD8E73EAF6443BC3F1C5A3FC5A037406D8A3D40"> : tensor<1x1x128xf32>
    %cst_371 = stablehlo.constant dense<9.99999997E-7> : tensor<1x3136x1xf32>
    %cst_372 = stablehlo.constant dense<0.000000e+00> : tensor<1x3136xf32>
    %cst_373 = stablehlo.constant dense<1.280000e+02> : tensor<1x3136xf32>
    %cst_374 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_375 = stablehlo.constant dense<"0x2DA5E6BCD748993DC229613D6C902E3FE8F1B93CC4E686BD3BEFC4BE1C22C53D87CFB2BC4A0EE9BB8AF79B3ED569F03C2DFC9C3D7A8A573D579C193D8D4B073F1B20C73D1109763EF321A5BD819AD5BE4F89EE3E05CC72BE3F62473ED3EC403D54F7BA3EC725A0BEDD35BA3DD6CE7B3CF63B3CBD37E9FBBDE9E37C3DAB4E4BBE9D7503BC4ACC423DA83810BDCEA8A73C5CDD7B3D6C7E9D3D2FE6DD3C03D0933E4767A4BE343505BD36C7F13D289FA8BDE9EB06BD23D41DBE8A6077BC4DD014BEBAA2B43D1203553D071BB7BC8901423E5FF6803DC60A0ABE095ECF3B36CD5A3D4A580B3EFBE3B73C1E69D5BEE49ADE3EC1E8393D83DF7B3C48AD4F3DD9D7B3BC09981C3D02C2AB3CD3F1DBBEC818C93BBC0CE4BE26ECC7BEF0F910BEF2AFDF3C53AF56BF9ACF69BDFEE8143C412AB43EAF55433D70B92B3F151412BE0AA2343D686B443E8AFE3BBDEAA1D4BEFC7D92BF13D75F3D4B6E62BDDD5410BD5AA0B13C50ED163DCCDE8A3DCF1C03BECCF5683E9C94173D1136E83E6376CDBEFE5F63BDE7091BBE20C7ACBD51BDEA3E169488397BBCA53DF03D98BC2B68823DDEA9153DD833293D05D20A3EDA6AC63D5BE651BB25AA0A3D528461BCBE06283D6C13CEBCAE3493BE02945FBF61C00BBD1C07713E82A8223F6713CFBEFD549CBDE6E56A3D44C4CB3C59C10B3DD576F23EAA372C3F2C30EB3D943CEE3E4BCFC13C95EF233D"> : tensor<1x1x1x128xf32>
    %cst_376 = stablehlo.constant dense_resource<__elided__> : tensor<4x4x3x128xf32>
    %0 = stablehlo.convolution(%arg0, %cst_376) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x224x224x3xf32>, tensor<4x4x3x128xf32>) -> tensor<1x56x56x128xf32>
    %1 = stablehlo.reshape %cst_375 : (tensor<1x1x1x128xf32>) -> tensor<1x128xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 3] : (tensor<1x128xf32>) -> tensor<1x56x56x128xf32>
    %3 = stablehlo.add %0, %2 : tensor<1x56x56x128xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x56x56x128xf32>) -> tensor<1x3136x128xf32>
    %5 = stablehlo.reduce(%4 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x3136x128xf32>, tensor<f32>) -> tensor<1x3136xf32>
    %6 = stablehlo.divide %5, %cst_373 : tensor<1x3136xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x1xf32>
    %8 = stablehlo.reshape %7 : (tensor<1x3136x1xf32>) -> tensor<1x3136xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x128xf32>
    %10 = stablehlo.subtract %4, %9 : tensor<1x3136x128xf32>
    %11 = stablehlo.multiply %4, %4 : tensor<1x3136x128xf32>
    %12 = stablehlo.reduce(%11 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x3136x128xf32>, tensor<f32>) -> tensor<1x3136xf32>
    %13 = stablehlo.divide %12, %cst_373 : tensor<1x3136xf32>
    %14 = stablehlo.multiply %6, %6 : tensor<1x3136xf32>
    %15 = stablehlo.subtract %13, %14 : tensor<1x3136xf32>
    %16 = stablehlo.maximum %cst_372, %15 : tensor<1x3136xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x1xf32>
    %18 = stablehlo.add %17, %cst_371 : tensor<1x3136x1xf32>
    %19 = stablehlo.rsqrt %18 : tensor<1x3136x1xf32>
    %20 = stablehlo.reshape %19 : (tensor<1x3136x1xf32>) -> tensor<1x3136xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x128xf32>
    %22 = stablehlo.reshape %cst_370 : (tensor<1x1x128xf32>) -> tensor<1x128xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 2] : (tensor<1x128xf32>) -> tensor<1x3136x128xf32>
    %24 = stablehlo.multiply %21, %23 : tensor<1x3136x128xf32>
    %25 = stablehlo.multiply %10, %24 : tensor<1x3136x128xf32>
    %26 = stablehlo.reshape %cst_369 : (tensor<1x1x128xf32>) -> tensor<1x128xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 2] : (tensor<1x128xf32>) -> tensor<1x3136x128xf32>
    %28 = stablehlo.add %25, %27 : tensor<1x3136x128xf32>
    %29 = stablehlo.reduce(%28 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x3136x128xf32>, tensor<f32>) -> tensor<1x3136xf32>
    %30 = stablehlo.divide %29, %cst_373 : tensor<1x3136xf32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x1xf32>
    %32 = stablehlo.reshape %31 : (tensor<1x3136x1xf32>) -> tensor<1x3136xf32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x128xf32>
    %34 = stablehlo.subtract %28, %33 : tensor<1x3136x128xf32>
    %35 = stablehlo.multiply %28, %28 : tensor<1x3136x128xf32>
    %36 = stablehlo.reduce(%35 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x3136x128xf32>, tensor<f32>) -> tensor<1x3136xf32>
    %37 = stablehlo.divide %36, %cst_373 : tensor<1x3136xf32>
    %38 = stablehlo.multiply %30, %30 : tensor<1x3136xf32>
    %39 = stablehlo.subtract %37, %38 : tensor<1x3136xf32>
    %40 = stablehlo.maximum %cst_372, %39 : tensor<1x3136xf32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x1xf32>
    %42 = stablehlo.add %41, %cst_371 : tensor<1x3136x1xf32>
    %43 = stablehlo.rsqrt %42 : tensor<1x3136x1xf32>
    %44 = stablehlo.reshape %43 : (tensor<1x3136x1xf32>) -> tensor<1x3136xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x128xf32>
    %46 = stablehlo.reshape %cst_368 : (tensor<1x1x128xf32>) -> tensor<1x128xf32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [0, 2] : (tensor<1x128xf32>) -> tensor<1x3136x128xf32>
    %48 = stablehlo.multiply %45, %47 : tensor<1x3136x128xf32>
    %49 = stablehlo.multiply %34, %48 : tensor<1x3136x128xf32>
    %50 = stablehlo.reshape %cst_367 : (tensor<1x1x128xf32>) -> tensor<1x128xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 2] : (tensor<1x128xf32>) -> tensor<1x3136x128xf32>
    %52 = stablehlo.add %49, %51 : tensor<1x3136x128xf32>
    %53 = stablehlo.reshape %52 : (tensor<1x3136x128xf32>) -> tensor<1x8x7x8x7x128xf32>
    %54 = stablehlo.transpose %53, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,8,8,7,7,128]{5,4,2,3,1,0}"} : (tensor<1x8x7x8x7x128xf32>) -> tensor<1x8x8x7x7x128xf32>
    %55 = stablehlo.reshape %54 : (tensor<1x8x8x7x7x128xf32>) -> tensor<64x49x128xf32>
    %56 = stablehlo.dot_general %55, %cst_366, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x49x128xf32>, tensor<128x384xf32>) -> tensor<64x49x384xf32>
    %57 = stablehlo.reshape %cst_365 : (tensor<1x1x384xf32>) -> tensor<384xf32>
    %58 = stablehlo.broadcast_in_dim %57, dims = [2] : (tensor<384xf32>) -> tensor<64x49x384xf32>
    %59 = stablehlo.add %56, %58 : tensor<64x49x384xf32>
    %60 = stablehlo.reshape %59 : (tensor<64x49x384xf32>) -> tensor<64x49x3x4x32xf32>
    %61 = stablehlo.transpose %60, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,64,4,49,32]{4,2,0,3,1}"} : (tensor<64x49x3x4x32xf32>) -> tensor<3x64x4x49x32xf32>
    %62 = stablehlo.slice %61 [0:1, 0:64, 0:4, 0:49, 0:32] : (tensor<3x64x4x49x32xf32>) -> tensor<1x64x4x49x32xf32>
    %63 = stablehlo.reshape %62 : (tensor<1x64x4x49x32xf32>) -> tensor<64x4x49x32xf32>
    %64 = stablehlo.multiply %63, %cst_364 : tensor<64x4x49x32xf32>
    %65 = stablehlo.slice %61 [1:2, 0:64, 0:4, 0:49, 0:32] : (tensor<3x64x4x49x32xf32>) -> tensor<1x64x4x49x32xf32>
    %66 = stablehlo.reshape %65 : (tensor<1x64x4x49x32xf32>) -> tensor<64x4x49x32xf32>
    %67 = stablehlo.transpose %66, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[64,4,32,49]{2,3,1,0}"} : (tensor<64x4x49x32xf32>) -> tensor<64x4x32x49xf32>
    %68 = stablehlo.dot_general %64, %67, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x4x49x32xf32>, tensor<64x4x32x49xf32>) -> tensor<64x4x49x49xf32>
    %69 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %70 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %71 = stablehlo.select %69, %70, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %73 = "stablehlo.gather"(%cst_363, %72) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4>}> : (tensor<169x4xf32>, tensor<2401x1xi32>) -> tensor<2401x4xf32>
    %74 = stablehlo.reshape %73 : (tensor<2401x4xf32>) -> tensor<49x49x4xf32>
    %75 = stablehlo.transpose %74, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[4,49,49]{0,2,1}"} : (tensor<49x49x4xf32>) -> tensor<4x49x49xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [1, 2, 3] : (tensor<4x49x49xf32>) -> tensor<1x4x49x49xf32>
    %77 = stablehlo.reshape %76 : (tensor<1x4x49x49xf32>) -> tensor<4x49x49xf32>
    %78 = stablehlo.broadcast_in_dim %77, dims = [1, 2, 3] : (tensor<4x49x49xf32>) -> tensor<64x4x49x49xf32>
    %79 = stablehlo.add %68, %78 : tensor<64x4x49x49xf32>
    %80 = stablehlo.reduce(%79 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<64x4x49x49xf32>, tensor<f32>) -> tensor<64x4x49xf32>
    %81 = stablehlo.broadcast_in_dim %80, dims = [0, 1, 2] : (tensor<64x4x49xf32>) -> tensor<64x4x49x1xf32>
    %82 = stablehlo.reshape %81 : (tensor<64x4x49x1xf32>) -> tensor<64x4x49xf32>
    %83 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2] : (tensor<64x4x49xf32>) -> tensor<64x4x49x49xf32>
    %84 = stablehlo.subtract %79, %83 : tensor<64x4x49x49xf32>
    %85 = stablehlo.exponential %84 : tensor<64x4x49x49xf32>
    %86 = stablehlo.reduce(%85 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<64x4x49x49xf32>, tensor<f32>) -> tensor<64x4x49xf32>
    %87 = stablehlo.broadcast_in_dim %86, dims = [0, 1, 2] : (tensor<64x4x49xf32>) -> tensor<64x4x49x1xf32>
    %88 = stablehlo.reshape %87 : (tensor<64x4x49x1xf32>) -> tensor<64x4x49xf32>
    %89 = stablehlo.broadcast_in_dim %88, dims = [0, 1, 2] : (tensor<64x4x49xf32>) -> tensor<64x4x49x49xf32>
    %90 = stablehlo.divide %85, %89 : tensor<64x4x49x49xf32>
    %91 = stablehlo.slice %61 [2:3, 0:64, 0:4, 0:49, 0:32] : (tensor<3x64x4x49x32xf32>) -> tensor<1x64x4x49x32xf32>
    %92 = stablehlo.reshape %91 : (tensor<1x64x4x49x32xf32>) -> tensor<64x4x49x32xf32>
    %93 = stablehlo.dot_general %90, %92, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x4x49x49xf32>, tensor<64x4x49x32xf32>) -> tensor<64x4x49x32xf32>
    %94 = stablehlo.transpose %93, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,49,4,32]{3,1,2,0}"} : (tensor<64x4x49x32xf32>) -> tensor<64x49x4x32xf32>
    %95 = stablehlo.reshape %94 : (tensor<64x49x4x32xf32>) -> tensor<64x49x128xf32>
    %96 = stablehlo.dot_general %95, %cst_358, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x49x128xf32>, tensor<128x128xf32>) -> tensor<64x49x128xf32>
    %97 = stablehlo.reshape %cst_357 : (tensor<1x1x128xf32>) -> tensor<128xf32>
    %98 = stablehlo.broadcast_in_dim %97, dims = [2] : (tensor<128xf32>) -> tensor<64x49x128xf32>
    %99 = stablehlo.add %96, %98 : tensor<64x49x128xf32>
    %100 = stablehlo.reshape %99 : (tensor<64x49x128xf32>) -> tensor<1x8x8x7x7x128xf32>
    %101 = stablehlo.transpose %100, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,8,7,8,7,128]{5,4,2,3,1,0}"} : (tensor<1x8x8x7x7x128xf32>) -> tensor<1x8x7x8x7x128xf32>
    %102 = stablehlo.reshape %101 : (tensor<1x8x7x8x7x128xf32>) -> tensor<1x3136x128xf32>
    %103 = stablehlo.add %28, %102 : tensor<1x3136x128xf32>
    %104 = stablehlo.reduce(%103 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x3136x128xf32>, tensor<f32>) -> tensor<1x3136xf32>
    %105 = stablehlo.divide %104, %cst_373 : tensor<1x3136xf32>
    %106 = stablehlo.broadcast_in_dim %105, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x1xf32>
    %107 = stablehlo.reshape %106 : (tensor<1x3136x1xf32>) -> tensor<1x3136xf32>
    %108 = stablehlo.broadcast_in_dim %107, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x128xf32>
    %109 = stablehlo.subtract %103, %108 : tensor<1x3136x128xf32>
    %110 = stablehlo.multiply %103, %103 : tensor<1x3136x128xf32>
    %111 = stablehlo.reduce(%110 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x3136x128xf32>, tensor<f32>) -> tensor<1x3136xf32>
    %112 = stablehlo.divide %111, %cst_373 : tensor<1x3136xf32>
    %113 = stablehlo.multiply %105, %105 : tensor<1x3136xf32>
    %114 = stablehlo.subtract %112, %113 : tensor<1x3136xf32>
    %115 = stablehlo.maximum %cst_372, %114 : tensor<1x3136xf32>
    %116 = stablehlo.broadcast_in_dim %115, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x1xf32>
    %117 = stablehlo.add %116, %cst_371 : tensor<1x3136x1xf32>
    %118 = stablehlo.rsqrt %117 : tensor<1x3136x1xf32>
    %119 = stablehlo.reshape %118 : (tensor<1x3136x1xf32>) -> tensor<1x3136xf32>
    %120 = stablehlo.broadcast_in_dim %119, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x128xf32>
    %121 = stablehlo.reshape %cst_356 : (tensor<1x1x128xf32>) -> tensor<1x128xf32>
    %122 = stablehlo.broadcast_in_dim %121, dims = [0, 2] : (tensor<1x128xf32>) -> tensor<1x3136x128xf32>
    %123 = stablehlo.multiply %120, %122 : tensor<1x3136x128xf32>
    %124 = stablehlo.multiply %109, %123 : tensor<1x3136x128xf32>
    %125 = stablehlo.reshape %cst_355 : (tensor<1x1x128xf32>) -> tensor<1x128xf32>
    %126 = stablehlo.broadcast_in_dim %125, dims = [0, 2] : (tensor<1x128xf32>) -> tensor<1x3136x128xf32>
    %127 = stablehlo.add %124, %126 : tensor<1x3136x128xf32>
    %128 = stablehlo.dot_general %127, %cst_354, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x3136x128xf32>, tensor<128x512xf32>) -> tensor<1x3136x512xf32>
    %129 = stablehlo.reshape %cst_353 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %130 = stablehlo.broadcast_in_dim %129, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x3136x512xf32>
    %131 = stablehlo.add %128, %130 : tensor<1x3136x512xf32>
    %132 = stablehlo.multiply %131, %131 : tensor<1x3136x512xf32>
    %133 = stablehlo.multiply %132, %131 : tensor<1x3136x512xf32>
    %134 = stablehlo.multiply %cst_349, %133 : tensor<1x3136x512xf32>
    %135 = stablehlo.add %131, %134 : tensor<1x3136x512xf32>
    %136 = stablehlo.multiply %cst_350, %135 : tensor<1x3136x512xf32>
    %137 = stablehlo.tanh %136 : tensor<1x3136x512xf32>
    %138 = stablehlo.add %cst_351, %137 : tensor<1x3136x512xf32>
    %139 = stablehlo.multiply %cst_352, %138 : tensor<1x3136x512xf32>
    %140 = stablehlo.multiply %131, %139 : tensor<1x3136x512xf32>
    %141 = stablehlo.dot_general %140, %cst_348, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x3136x512xf32>, tensor<512x128xf32>) -> tensor<1x3136x128xf32>
    %142 = stablehlo.reshape %cst_347 : (tensor<1x1x128xf32>) -> tensor<1x128xf32>
    %143 = stablehlo.broadcast_in_dim %142, dims = [0, 2] : (tensor<1x128xf32>) -> tensor<1x3136x128xf32>
    %144 = stablehlo.add %141, %143 : tensor<1x3136x128xf32>
    %145 = stablehlo.add %103, %144 : tensor<1x3136x128xf32>
    %146 = stablehlo.reduce(%145 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x3136x128xf32>, tensor<f32>) -> tensor<1x3136xf32>
    %147 = stablehlo.divide %146, %cst_373 : tensor<1x3136xf32>
    %148 = stablehlo.broadcast_in_dim %147, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x1xf32>
    %149 = stablehlo.reshape %148 : (tensor<1x3136x1xf32>) -> tensor<1x3136xf32>
    %150 = stablehlo.broadcast_in_dim %149, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x128xf32>
    %151 = stablehlo.subtract %145, %150 : tensor<1x3136x128xf32>
    %152 = stablehlo.multiply %145, %145 : tensor<1x3136x128xf32>
    %153 = stablehlo.reduce(%152 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x3136x128xf32>, tensor<f32>) -> tensor<1x3136xf32>
    %154 = stablehlo.divide %153, %cst_373 : tensor<1x3136xf32>
    %155 = stablehlo.multiply %147, %147 : tensor<1x3136xf32>
    %156 = stablehlo.subtract %154, %155 : tensor<1x3136xf32>
    %157 = stablehlo.maximum %cst_372, %156 : tensor<1x3136xf32>
    %158 = stablehlo.broadcast_in_dim %157, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x1xf32>
    %159 = stablehlo.add %158, %cst_371 : tensor<1x3136x1xf32>
    %160 = stablehlo.rsqrt %159 : tensor<1x3136x1xf32>
    %161 = stablehlo.reshape %160 : (tensor<1x3136x1xf32>) -> tensor<1x3136xf32>
    %162 = stablehlo.broadcast_in_dim %161, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x128xf32>
    %163 = stablehlo.reshape %cst_346 : (tensor<1x1x128xf32>) -> tensor<1x128xf32>
    %164 = stablehlo.broadcast_in_dim %163, dims = [0, 2] : (tensor<1x128xf32>) -> tensor<1x3136x128xf32>
    %165 = stablehlo.multiply %162, %164 : tensor<1x3136x128xf32>
    %166 = stablehlo.multiply %151, %165 : tensor<1x3136x128xf32>
    %167 = stablehlo.reshape %cst_345 : (tensor<1x1x128xf32>) -> tensor<1x128xf32>
    %168 = stablehlo.broadcast_in_dim %167, dims = [0, 2] : (tensor<1x128xf32>) -> tensor<1x3136x128xf32>
    %169 = stablehlo.add %166, %168 : tensor<1x3136x128xf32>
    %170 = stablehlo.reshape %169 : (tensor<1x3136x128xf32>) -> tensor<1x56x56x128xf32>
    %171 = stablehlo.slice %170 [0:1, 3:56, 0:56, 0:128] : (tensor<1x56x56x128xf32>) -> tensor<1x53x56x128xf32>
    %172 = stablehlo.slice %170 [0:1, 0:3, 0:56, 0:128] : (tensor<1x56x56x128xf32>) -> tensor<1x3x56x128xf32>
    %173 = stablehlo.concatenate %171, %172, dim = 1 : (tensor<1x53x56x128xf32>, tensor<1x3x56x128xf32>) -> tensor<1x56x56x128xf32>
    %174 = stablehlo.slice %173 [0:1, 0:56, 3:56, 0:128] : (tensor<1x56x56x128xf32>) -> tensor<1x56x53x128xf32>
    %175 = stablehlo.slice %173 [0:1, 0:56, 0:3, 0:128] : (tensor<1x56x56x128xf32>) -> tensor<1x56x3x128xf32>
    %176 = stablehlo.concatenate %174, %175, dim = 2 : (tensor<1x56x53x128xf32>, tensor<1x56x3x128xf32>) -> tensor<1x56x56x128xf32>
    %177 = stablehlo.reshape %176 : (tensor<1x56x56x128xf32>) -> tensor<1x8x7x8x7x128xf32>
    %178 = stablehlo.transpose %177, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,8,8,7,7,128]{5,4,2,3,1,0}"} : (tensor<1x8x7x8x7x128xf32>) -> tensor<1x8x8x7x7x128xf32>
    %179 = stablehlo.reshape %178 : (tensor<1x8x8x7x7x128xf32>) -> tensor<64x49x128xf32>
    %180 = stablehlo.dot_general %179, %cst_344, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x49x128xf32>, tensor<128x384xf32>) -> tensor<64x49x384xf32>
    %181 = stablehlo.reshape %cst_343 : (tensor<1x1x384xf32>) -> tensor<384xf32>
    %182 = stablehlo.broadcast_in_dim %181, dims = [2] : (tensor<384xf32>) -> tensor<64x49x384xf32>
    %183 = stablehlo.add %180, %182 : tensor<64x49x384xf32>
    %184 = stablehlo.reshape %183 : (tensor<64x49x384xf32>) -> tensor<64x49x3x4x32xf32>
    %185 = stablehlo.transpose %184, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,64,4,49,32]{4,2,0,3,1}"} : (tensor<64x49x3x4x32xf32>) -> tensor<3x64x4x49x32xf32>
    %186 = stablehlo.slice %185 [0:1, 0:64, 0:4, 0:49, 0:32] : (tensor<3x64x4x49x32xf32>) -> tensor<1x64x4x49x32xf32>
    %187 = stablehlo.reshape %186 : (tensor<1x64x4x49x32xf32>) -> tensor<64x4x49x32xf32>
    %188 = stablehlo.multiply %187, %cst_364 : tensor<64x4x49x32xf32>
    %189 = stablehlo.slice %185 [1:2, 0:64, 0:4, 0:49, 0:32] : (tensor<3x64x4x49x32xf32>) -> tensor<1x64x4x49x32xf32>
    %190 = stablehlo.reshape %189 : (tensor<1x64x4x49x32xf32>) -> tensor<64x4x49x32xf32>
    %191 = stablehlo.transpose %190, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[64,4,32,49]{2,3,1,0}"} : (tensor<64x4x49x32xf32>) -> tensor<64x4x32x49xf32>
    %192 = stablehlo.dot_general %188, %191, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x4x49x32xf32>, tensor<64x4x32x49xf32>) -> tensor<64x4x49x49xf32>
    %193 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %194 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %195 = stablehlo.select %193, %194, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %196 = stablehlo.broadcast_in_dim %195, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %197 = "stablehlo.gather"(%cst_342, %196) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4>}> : (tensor<169x4xf32>, tensor<2401x1xi32>) -> tensor<2401x4xf32>
    %198 = stablehlo.reshape %197 : (tensor<2401x4xf32>) -> tensor<49x49x4xf32>
    %199 = stablehlo.transpose %198, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[4,49,49]{0,2,1}"} : (tensor<49x49x4xf32>) -> tensor<4x49x49xf32>
    %200 = stablehlo.broadcast_in_dim %199, dims = [1, 2, 3] : (tensor<4x49x49xf32>) -> tensor<1x4x49x49xf32>
    %201 = stablehlo.reshape %200 : (tensor<1x4x49x49xf32>) -> tensor<4x49x49xf32>
    %202 = stablehlo.broadcast_in_dim %201, dims = [1, 2, 3] : (tensor<4x49x49xf32>) -> tensor<64x4x49x49xf32>
    %203 = stablehlo.add %192, %202 : tensor<64x4x49x49xf32>
    %204 = stablehlo.reduce(%203 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<64x4x49x49xf32>, tensor<f32>) -> tensor<64x4x49xf32>
    %205 = stablehlo.broadcast_in_dim %204, dims = [0, 1, 2] : (tensor<64x4x49xf32>) -> tensor<64x4x49x1xf32>
    %206 = stablehlo.reshape %205 : (tensor<64x4x49x1xf32>) -> tensor<64x4x49xf32>
    %207 = stablehlo.broadcast_in_dim %206, dims = [0, 1, 2] : (tensor<64x4x49xf32>) -> tensor<64x4x49x49xf32>
    %208 = stablehlo.subtract %203, %207 : tensor<64x4x49x49xf32>
    %209 = stablehlo.exponential %208 : tensor<64x4x49x49xf32>
    %210 = stablehlo.reduce(%209 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<64x4x49x49xf32>, tensor<f32>) -> tensor<64x4x49xf32>
    %211 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2] : (tensor<64x4x49xf32>) -> tensor<64x4x49x1xf32>
    %212 = stablehlo.reshape %211 : (tensor<64x4x49x1xf32>) -> tensor<64x4x49xf32>
    %213 = stablehlo.broadcast_in_dim %212, dims = [0, 1, 2] : (tensor<64x4x49xf32>) -> tensor<64x4x49x49xf32>
    %214 = stablehlo.divide %209, %213 : tensor<64x4x49x49xf32>
    %215 = stablehlo.slice %185 [2:3, 0:64, 0:4, 0:49, 0:32] : (tensor<3x64x4x49x32xf32>) -> tensor<1x64x4x49x32xf32>
    %216 = stablehlo.reshape %215 : (tensor<1x64x4x49x32xf32>) -> tensor<64x4x49x32xf32>
    %217 = stablehlo.dot_general %214, %216, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x4x49x49xf32>, tensor<64x4x49x32xf32>) -> tensor<64x4x49x32xf32>
    %218 = stablehlo.transpose %217, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,49,4,32]{3,1,2,0}"} : (tensor<64x4x49x32xf32>) -> tensor<64x49x4x32xf32>
    %219 = stablehlo.reshape %218 : (tensor<64x49x4x32xf32>) -> tensor<64x49x128xf32>
    %220 = stablehlo.dot_general %219, %cst_341, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x49x128xf32>, tensor<128x128xf32>) -> tensor<64x49x128xf32>
    %221 = stablehlo.reshape %cst_340 : (tensor<1x1x128xf32>) -> tensor<128xf32>
    %222 = stablehlo.broadcast_in_dim %221, dims = [2] : (tensor<128xf32>) -> tensor<64x49x128xf32>
    %223 = stablehlo.add %220, %222 : tensor<64x49x128xf32>
    %224 = stablehlo.reshape %223 : (tensor<64x49x128xf32>) -> tensor<1x8x8x7x7x128xf32>
    %225 = stablehlo.transpose %224, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,8,7,8,7,128]{5,4,2,3,1,0}"} : (tensor<1x8x8x7x7x128xf32>) -> tensor<1x8x7x8x7x128xf32>
    %226 = stablehlo.reshape %225 : (tensor<1x8x7x8x7x128xf32>) -> tensor<1x56x56x128xf32>
    %227 = stablehlo.slice %226 [0:1, 53:56, 0:56, 0:128] : (tensor<1x56x56x128xf32>) -> tensor<1x3x56x128xf32>
    %228 = stablehlo.slice %226 [0:1, 0:53, 0:56, 0:128] : (tensor<1x56x56x128xf32>) -> tensor<1x53x56x128xf32>
    %229 = stablehlo.concatenate %227, %228, dim = 1 : (tensor<1x3x56x128xf32>, tensor<1x53x56x128xf32>) -> tensor<1x56x56x128xf32>
    %230 = stablehlo.slice %229 [0:1, 0:56, 53:56, 0:128] : (tensor<1x56x56x128xf32>) -> tensor<1x56x3x128xf32>
    %231 = stablehlo.slice %229 [0:1, 0:56, 0:53, 0:128] : (tensor<1x56x56x128xf32>) -> tensor<1x56x53x128xf32>
    %232 = stablehlo.concatenate %230, %231, dim = 2 : (tensor<1x56x3x128xf32>, tensor<1x56x53x128xf32>) -> tensor<1x56x56x128xf32>
    %233 = stablehlo.reshape %232 : (tensor<1x56x56x128xf32>) -> tensor<1x3136x128xf32>
    %234 = stablehlo.add %145, %233 : tensor<1x3136x128xf32>
    %235 = stablehlo.reduce(%234 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x3136x128xf32>, tensor<f32>) -> tensor<1x3136xf32>
    %236 = stablehlo.divide %235, %cst_373 : tensor<1x3136xf32>
    %237 = stablehlo.broadcast_in_dim %236, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x1xf32>
    %238 = stablehlo.reshape %237 : (tensor<1x3136x1xf32>) -> tensor<1x3136xf32>
    %239 = stablehlo.broadcast_in_dim %238, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x128xf32>
    %240 = stablehlo.subtract %234, %239 : tensor<1x3136x128xf32>
    %241 = stablehlo.multiply %234, %234 : tensor<1x3136x128xf32>
    %242 = stablehlo.reduce(%241 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x3136x128xf32>, tensor<f32>) -> tensor<1x3136xf32>
    %243 = stablehlo.divide %242, %cst_373 : tensor<1x3136xf32>
    %244 = stablehlo.multiply %236, %236 : tensor<1x3136xf32>
    %245 = stablehlo.subtract %243, %244 : tensor<1x3136xf32>
    %246 = stablehlo.maximum %cst_372, %245 : tensor<1x3136xf32>
    %247 = stablehlo.broadcast_in_dim %246, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x1xf32>
    %248 = stablehlo.add %247, %cst_371 : tensor<1x3136x1xf32>
    %249 = stablehlo.rsqrt %248 : tensor<1x3136x1xf32>
    %250 = stablehlo.reshape %249 : (tensor<1x3136x1xf32>) -> tensor<1x3136xf32>
    %251 = stablehlo.broadcast_in_dim %250, dims = [0, 1] : (tensor<1x3136xf32>) -> tensor<1x3136x128xf32>
    %252 = stablehlo.reshape %cst_339 : (tensor<1x1x128xf32>) -> tensor<1x128xf32>
    %253 = stablehlo.broadcast_in_dim %252, dims = [0, 2] : (tensor<1x128xf32>) -> tensor<1x3136x128xf32>
    %254 = stablehlo.multiply %251, %253 : tensor<1x3136x128xf32>
    %255 = stablehlo.multiply %240, %254 : tensor<1x3136x128xf32>
    %256 = stablehlo.reshape %cst_338 : (tensor<1x1x128xf32>) -> tensor<1x128xf32>
    %257 = stablehlo.broadcast_in_dim %256, dims = [0, 2] : (tensor<1x128xf32>) -> tensor<1x3136x128xf32>
    %258 = stablehlo.add %255, %257 : tensor<1x3136x128xf32>
    %259 = stablehlo.dot_general %258, %cst_337, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x3136x128xf32>, tensor<128x512xf32>) -> tensor<1x3136x512xf32>
    %260 = stablehlo.reshape %cst_336 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %261 = stablehlo.broadcast_in_dim %260, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x3136x512xf32>
    %262 = stablehlo.add %259, %261 : tensor<1x3136x512xf32>
    %263 = stablehlo.multiply %262, %262 : tensor<1x3136x512xf32>
    %264 = stablehlo.multiply %263, %262 : tensor<1x3136x512xf32>
    %265 = stablehlo.multiply %cst_349, %264 : tensor<1x3136x512xf32>
    %266 = stablehlo.add %262, %265 : tensor<1x3136x512xf32>
    %267 = stablehlo.multiply %cst_350, %266 : tensor<1x3136x512xf32>
    %268 = stablehlo.tanh %267 : tensor<1x3136x512xf32>
    %269 = stablehlo.add %cst_351, %268 : tensor<1x3136x512xf32>
    %270 = stablehlo.multiply %cst_352, %269 : tensor<1x3136x512xf32>
    %271 = stablehlo.multiply %262, %270 : tensor<1x3136x512xf32>
    %272 = stablehlo.dot_general %271, %cst_335, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x3136x512xf32>, tensor<512x128xf32>) -> tensor<1x3136x128xf32>
    %273 = stablehlo.reshape %cst_334 : (tensor<1x1x128xf32>) -> tensor<1x128xf32>
    %274 = stablehlo.broadcast_in_dim %273, dims = [0, 2] : (tensor<1x128xf32>) -> tensor<1x3136x128xf32>
    %275 = stablehlo.add %272, %274 : tensor<1x3136x128xf32>
    %276 = stablehlo.add %234, %275 : tensor<1x3136x128xf32>
    %277 = stablehlo.reshape %276 : (tensor<1x3136x128xf32>) -> tensor<1x56x56x128xf32>
    %278 = stablehlo.iota dim = 0 : tensor<28xi32>
    %279 = stablehlo.multiply %c_332, %278 : tensor<28xi32>
    %280 = stablehlo.add %c_333, %279 : tensor<28xi32>
    %281 = stablehlo.broadcast_in_dim %280, dims = [0] : (tensor<28xi32>) -> tensor<28x28x1xi32>
    %282 = stablehlo.iota dim = 0 : tensor<28xi32>
    %283 = stablehlo.multiply %c_332, %282 : tensor<28xi32>
    %284 = stablehlo.add %c_333, %283 : tensor<28xi32>
    %285 = stablehlo.broadcast_in_dim %284, dims = [1] : (tensor<28xi32>) -> tensor<28x28x1xi32>
    %286 = stablehlo.concatenate %281, %285, dim = 2 : (tensor<28x28x1xi32>, tensor<28x28x1xi32>) -> tensor<28x28x2xi32>
    %287 = "stablehlo.gather"(%277, %286) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 3], collapsed_slice_dims = [1, 2], start_index_map = [1, 2], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 1, 128>}> : (tensor<1x56x56x128xf32>, tensor<28x28x2xi32>) -> tensor<1x28x28x128xf32>
    %288 = stablehlo.iota dim = 0 : tensor<28xi32>
    %289 = stablehlo.multiply %c_332, %288 : tensor<28xi32>
    %290 = stablehlo.add %c_331, %289 : tensor<28xi32>
    %291 = stablehlo.broadcast_in_dim %290, dims = [0] : (tensor<28xi32>) -> tensor<28x28x1xi32>
    %292 = stablehlo.iota dim = 0 : tensor<28xi32>
    %293 = stablehlo.multiply %c_332, %292 : tensor<28xi32>
    %294 = stablehlo.add %c_333, %293 : tensor<28xi32>
    %295 = stablehlo.broadcast_in_dim %294, dims = [1] : (tensor<28xi32>) -> tensor<28x28x1xi32>
    %296 = stablehlo.concatenate %291, %295, dim = 2 : (tensor<28x28x1xi32>, tensor<28x28x1xi32>) -> tensor<28x28x2xi32>
    %297 = "stablehlo.gather"(%277, %296) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 3], collapsed_slice_dims = [1, 2], start_index_map = [1, 2], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 1, 128>}> : (tensor<1x56x56x128xf32>, tensor<28x28x2xi32>) -> tensor<1x28x28x128xf32>
    %298 = stablehlo.iota dim = 0 : tensor<28xi32>
    %299 = stablehlo.multiply %c_332, %298 : tensor<28xi32>
    %300 = stablehlo.add %c_333, %299 : tensor<28xi32>
    %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<28xi32>) -> tensor<28x28x1xi32>
    %302 = stablehlo.iota dim = 0 : tensor<28xi32>
    %303 = stablehlo.multiply %c_332, %302 : tensor<28xi32>
    %304 = stablehlo.add %c_331, %303 : tensor<28xi32>
    %305 = stablehlo.broadcast_in_dim %304, dims = [1] : (tensor<28xi32>) -> tensor<28x28x1xi32>
    %306 = stablehlo.concatenate %301, %305, dim = 2 : (tensor<28x28x1xi32>, tensor<28x28x1xi32>) -> tensor<28x28x2xi32>
    %307 = "stablehlo.gather"(%277, %306) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 3], collapsed_slice_dims = [1, 2], start_index_map = [1, 2], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 1, 128>}> : (tensor<1x56x56x128xf32>, tensor<28x28x2xi32>) -> tensor<1x28x28x128xf32>
    %308 = stablehlo.iota dim = 0 : tensor<28xi32>
    %309 = stablehlo.multiply %c_332, %308 : tensor<28xi32>
    %310 = stablehlo.add %c_331, %309 : tensor<28xi32>
    %311 = stablehlo.broadcast_in_dim %310, dims = [0] : (tensor<28xi32>) -> tensor<28x28x1xi32>
    %312 = stablehlo.iota dim = 0 : tensor<28xi32>
    %313 = stablehlo.multiply %c_332, %312 : tensor<28xi32>
    %314 = stablehlo.add %c_331, %313 : tensor<28xi32>
    %315 = stablehlo.broadcast_in_dim %314, dims = [1] : (tensor<28xi32>) -> tensor<28x28x1xi32>
    %316 = stablehlo.concatenate %311, %315, dim = 2 : (tensor<28x28x1xi32>, tensor<28x28x1xi32>) -> tensor<28x28x2xi32>
    %317 = "stablehlo.gather"(%277, %316) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 3], collapsed_slice_dims = [1, 2], start_index_map = [1, 2], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 1, 128>}> : (tensor<1x56x56x128xf32>, tensor<28x28x2xi32>) -> tensor<1x28x28x128xf32>
    %318 = stablehlo.concatenate %287, %297, %307, %317, dim = 3 : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x512xf32>
    %319 = stablehlo.reshape %318 : (tensor<1x28x28x512xf32>) -> tensor<1x784x512xf32>
    %320 = stablehlo.reduce(%319 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x784x512xf32>, tensor<f32>) -> tensor<1x784xf32>
    %321 = stablehlo.divide %320, %cst_330 : tensor<1x784xf32>
    %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x1xf32>
    %323 = stablehlo.reshape %322 : (tensor<1x784x1xf32>) -> tensor<1x784xf32>
    %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x512xf32>
    %325 = stablehlo.subtract %319, %324 : tensor<1x784x512xf32>
    %326 = stablehlo.multiply %319, %319 : tensor<1x784x512xf32>
    %327 = stablehlo.reduce(%326 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x784x512xf32>, tensor<f32>) -> tensor<1x784xf32>
    %328 = stablehlo.divide %327, %cst_330 : tensor<1x784xf32>
    %329 = stablehlo.multiply %321, %321 : tensor<1x784xf32>
    %330 = stablehlo.subtract %328, %329 : tensor<1x784xf32>
    %331 = stablehlo.maximum %cst_329, %330 : tensor<1x784xf32>
    %332 = stablehlo.broadcast_in_dim %331, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x1xf32>
    %333 = stablehlo.add %332, %cst_328 : tensor<1x784x1xf32>
    %334 = stablehlo.rsqrt %333 : tensor<1x784x1xf32>
    %335 = stablehlo.reshape %334 : (tensor<1x784x1xf32>) -> tensor<1x784xf32>
    %336 = stablehlo.broadcast_in_dim %335, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x512xf32>
    %337 = stablehlo.reshape %cst_327 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %338 = stablehlo.broadcast_in_dim %337, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x784x512xf32>
    %339 = stablehlo.multiply %336, %338 : tensor<1x784x512xf32>
    %340 = stablehlo.multiply %325, %339 : tensor<1x784x512xf32>
    %341 = stablehlo.reshape %cst_326 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %342 = stablehlo.broadcast_in_dim %341, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x784x512xf32>
    %343 = stablehlo.add %340, %342 : tensor<1x784x512xf32>
    %344 = stablehlo.dot_general %343, %cst_325, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x784x512xf32>, tensor<512x256xf32>) -> tensor<1x784x256xf32>
    %345 = stablehlo.reduce(%344 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x784x256xf32>, tensor<f32>) -> tensor<1x784xf32>
    %346 = stablehlo.divide %345, %cst_324 : tensor<1x784xf32>
    %347 = stablehlo.broadcast_in_dim %346, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x1xf32>
    %348 = stablehlo.reshape %347 : (tensor<1x784x1xf32>) -> tensor<1x784xf32>
    %349 = stablehlo.broadcast_in_dim %348, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x256xf32>
    %350 = stablehlo.subtract %344, %349 : tensor<1x784x256xf32>
    %351 = stablehlo.multiply %344, %344 : tensor<1x784x256xf32>
    %352 = stablehlo.reduce(%351 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x784x256xf32>, tensor<f32>) -> tensor<1x784xf32>
    %353 = stablehlo.divide %352, %cst_324 : tensor<1x784xf32>
    %354 = stablehlo.multiply %346, %346 : tensor<1x784xf32>
    %355 = stablehlo.subtract %353, %354 : tensor<1x784xf32>
    %356 = stablehlo.maximum %cst_329, %355 : tensor<1x784xf32>
    %357 = stablehlo.broadcast_in_dim %356, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x1xf32>
    %358 = stablehlo.add %357, %cst_328 : tensor<1x784x1xf32>
    %359 = stablehlo.rsqrt %358 : tensor<1x784x1xf32>
    %360 = stablehlo.reshape %359 : (tensor<1x784x1xf32>) -> tensor<1x784xf32>
    %361 = stablehlo.broadcast_in_dim %360, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x256xf32>
    %362 = stablehlo.reshape %cst_323 : (tensor<1x1x256xf32>) -> tensor<1x256xf32>
    %363 = stablehlo.broadcast_in_dim %362, dims = [0, 2] : (tensor<1x256xf32>) -> tensor<1x784x256xf32>
    %364 = stablehlo.multiply %361, %363 : tensor<1x784x256xf32>
    %365 = stablehlo.multiply %350, %364 : tensor<1x784x256xf32>
    %366 = stablehlo.reshape %cst_322 : (tensor<1x1x256xf32>) -> tensor<1x256xf32>
    %367 = stablehlo.broadcast_in_dim %366, dims = [0, 2] : (tensor<1x256xf32>) -> tensor<1x784x256xf32>
    %368 = stablehlo.add %365, %367 : tensor<1x784x256xf32>
    %369 = stablehlo.reshape %368 : (tensor<1x784x256xf32>) -> tensor<1x4x7x4x7x256xf32>
    %370 = stablehlo.transpose %369, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,4,4,7,7,256]{5,4,2,3,1,0}"} : (tensor<1x4x7x4x7x256xf32>) -> tensor<1x4x4x7x7x256xf32>
    %371 = stablehlo.reshape %370 : (tensor<1x4x4x7x7x256xf32>) -> tensor<16x49x256xf32>
    %372 = stablehlo.dot_general %371, %cst_321, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x49x256xf32>, tensor<256x768xf32>) -> tensor<16x49x768xf32>
    %373 = stablehlo.reshape %cst_320 : (tensor<1x1x768xf32>) -> tensor<768xf32>
    %374 = stablehlo.broadcast_in_dim %373, dims = [2] : (tensor<768xf32>) -> tensor<16x49x768xf32>
    %375 = stablehlo.add %372, %374 : tensor<16x49x768xf32>
    %376 = stablehlo.reshape %375 : (tensor<16x49x768xf32>) -> tensor<16x49x3x8x32xf32>
    %377 = stablehlo.transpose %376, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,16,8,49,32]{4,2,0,3,1}"} : (tensor<16x49x3x8x32xf32>) -> tensor<3x16x8x49x32xf32>
    %378 = stablehlo.slice %377 [0:1, 0:16, 0:8, 0:49, 0:32] : (tensor<3x16x8x49x32xf32>) -> tensor<1x16x8x49x32xf32>
    %379 = stablehlo.reshape %378 : (tensor<1x16x8x49x32xf32>) -> tensor<16x8x49x32xf32>
    %380 = stablehlo.multiply %379, %cst_319 : tensor<16x8x49x32xf32>
    %381 = stablehlo.slice %377 [1:2, 0:16, 0:8, 0:49, 0:32] : (tensor<3x16x8x49x32xf32>) -> tensor<1x16x8x49x32xf32>
    %382 = stablehlo.reshape %381 : (tensor<1x16x8x49x32xf32>) -> tensor<16x8x49x32xf32>
    %383 = stablehlo.transpose %382, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[16,8,32,49]{2,3,1,0}"} : (tensor<16x8x49x32xf32>) -> tensor<16x8x32x49xf32>
    %384 = stablehlo.dot_general %380, %383, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<16x8x49x32xf32>, tensor<16x8x32x49xf32>) -> tensor<16x8x49x49xf32>
    %385 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %386 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %387 = stablehlo.select %385, %386, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %388 = stablehlo.broadcast_in_dim %387, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %389 = "stablehlo.gather"(%cst_318, %388) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 8>}> : (tensor<169x8xf32>, tensor<2401x1xi32>) -> tensor<2401x8xf32>
    %390 = stablehlo.reshape %389 : (tensor<2401x8xf32>) -> tensor<49x49x8xf32>
    %391 = stablehlo.transpose %390, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,49,49]{0,2,1}"} : (tensor<49x49x8xf32>) -> tensor<8x49x49xf32>
    %392 = stablehlo.broadcast_in_dim %391, dims = [1, 2, 3] : (tensor<8x49x49xf32>) -> tensor<1x8x49x49xf32>
    %393 = stablehlo.reshape %392 : (tensor<1x8x49x49xf32>) -> tensor<8x49x49xf32>
    %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2, 3] : (tensor<8x49x49xf32>) -> tensor<16x8x49x49xf32>
    %395 = stablehlo.add %384, %394 : tensor<16x8x49x49xf32>
    %396 = stablehlo.reduce(%395 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<16x8x49x49xf32>, tensor<f32>) -> tensor<16x8x49xf32>
    %397 = stablehlo.broadcast_in_dim %396, dims = [0, 1, 2] : (tensor<16x8x49xf32>) -> tensor<16x8x49x1xf32>
    %398 = stablehlo.reshape %397 : (tensor<16x8x49x1xf32>) -> tensor<16x8x49xf32>
    %399 = stablehlo.broadcast_in_dim %398, dims = [0, 1, 2] : (tensor<16x8x49xf32>) -> tensor<16x8x49x49xf32>
    %400 = stablehlo.subtract %395, %399 : tensor<16x8x49x49xf32>
    %401 = stablehlo.exponential %400 : tensor<16x8x49x49xf32>
    %402 = stablehlo.reduce(%401 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<16x8x49x49xf32>, tensor<f32>) -> tensor<16x8x49xf32>
    %403 = stablehlo.broadcast_in_dim %402, dims = [0, 1, 2] : (tensor<16x8x49xf32>) -> tensor<16x8x49x1xf32>
    %404 = stablehlo.reshape %403 : (tensor<16x8x49x1xf32>) -> tensor<16x8x49xf32>
    %405 = stablehlo.broadcast_in_dim %404, dims = [0, 1, 2] : (tensor<16x8x49xf32>) -> tensor<16x8x49x49xf32>
    %406 = stablehlo.divide %401, %405 : tensor<16x8x49x49xf32>
    %407 = stablehlo.slice %377 [2:3, 0:16, 0:8, 0:49, 0:32] : (tensor<3x16x8x49x32xf32>) -> tensor<1x16x8x49x32xf32>
    %408 = stablehlo.reshape %407 : (tensor<1x16x8x49x32xf32>) -> tensor<16x8x49x32xf32>
    %409 = stablehlo.dot_general %406, %408, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<16x8x49x49xf32>, tensor<16x8x49x32xf32>) -> tensor<16x8x49x32xf32>
    %410 = stablehlo.transpose %409, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,49,8,32]{3,1,2,0}"} : (tensor<16x8x49x32xf32>) -> tensor<16x49x8x32xf32>
    %411 = stablehlo.reshape %410 : (tensor<16x49x8x32xf32>) -> tensor<16x49x256xf32>
    %412 = stablehlo.dot_general %411, %cst_317, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x49x256xf32>, tensor<256x256xf32>) -> tensor<16x49x256xf32>
    %413 = stablehlo.reshape %cst_316 : (tensor<1x1x256xf32>) -> tensor<256xf32>
    %414 = stablehlo.broadcast_in_dim %413, dims = [2] : (tensor<256xf32>) -> tensor<16x49x256xf32>
    %415 = stablehlo.add %412, %414 : tensor<16x49x256xf32>
    %416 = stablehlo.reshape %415 : (tensor<16x49x256xf32>) -> tensor<1x4x4x7x7x256xf32>
    %417 = stablehlo.transpose %416, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,4,7,4,7,256]{5,4,2,3,1,0}"} : (tensor<1x4x4x7x7x256xf32>) -> tensor<1x4x7x4x7x256xf32>
    %418 = stablehlo.reshape %417 : (tensor<1x4x7x4x7x256xf32>) -> tensor<1x784x256xf32>
    %419 = stablehlo.add %344, %418 : tensor<1x784x256xf32>
    %420 = stablehlo.reduce(%419 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x784x256xf32>, tensor<f32>) -> tensor<1x784xf32>
    %421 = stablehlo.divide %420, %cst_324 : tensor<1x784xf32>
    %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x1xf32>
    %423 = stablehlo.reshape %422 : (tensor<1x784x1xf32>) -> tensor<1x784xf32>
    %424 = stablehlo.broadcast_in_dim %423, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x256xf32>
    %425 = stablehlo.subtract %419, %424 : tensor<1x784x256xf32>
    %426 = stablehlo.multiply %419, %419 : tensor<1x784x256xf32>
    %427 = stablehlo.reduce(%426 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x784x256xf32>, tensor<f32>) -> tensor<1x784xf32>
    %428 = stablehlo.divide %427, %cst_324 : tensor<1x784xf32>
    %429 = stablehlo.multiply %421, %421 : tensor<1x784xf32>
    %430 = stablehlo.subtract %428, %429 : tensor<1x784xf32>
    %431 = stablehlo.maximum %cst_329, %430 : tensor<1x784xf32>
    %432 = stablehlo.broadcast_in_dim %431, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x1xf32>
    %433 = stablehlo.add %432, %cst_328 : tensor<1x784x1xf32>
    %434 = stablehlo.rsqrt %433 : tensor<1x784x1xf32>
    %435 = stablehlo.reshape %434 : (tensor<1x784x1xf32>) -> tensor<1x784xf32>
    %436 = stablehlo.broadcast_in_dim %435, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x256xf32>
    %437 = stablehlo.reshape %cst_315 : (tensor<1x1x256xf32>) -> tensor<1x256xf32>
    %438 = stablehlo.broadcast_in_dim %437, dims = [0, 2] : (tensor<1x256xf32>) -> tensor<1x784x256xf32>
    %439 = stablehlo.multiply %436, %438 : tensor<1x784x256xf32>
    %440 = stablehlo.multiply %425, %439 : tensor<1x784x256xf32>
    %441 = stablehlo.reshape %cst_314 : (tensor<1x1x256xf32>) -> tensor<1x256xf32>
    %442 = stablehlo.broadcast_in_dim %441, dims = [0, 2] : (tensor<1x256xf32>) -> tensor<1x784x256xf32>
    %443 = stablehlo.add %440, %442 : tensor<1x784x256xf32>
    %444 = stablehlo.dot_general %443, %cst_313, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x784x256xf32>, tensor<256x1024xf32>) -> tensor<1x784x1024xf32>
    %445 = stablehlo.reshape %cst_312 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %446 = stablehlo.broadcast_in_dim %445, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x784x1024xf32>
    %447 = stablehlo.add %444, %446 : tensor<1x784x1024xf32>
    %448 = stablehlo.multiply %447, %447 : tensor<1x784x1024xf32>
    %449 = stablehlo.multiply %448, %447 : tensor<1x784x1024xf32>
    %450 = stablehlo.multiply %cst_308, %449 : tensor<1x784x1024xf32>
    %451 = stablehlo.add %447, %450 : tensor<1x784x1024xf32>
    %452 = stablehlo.multiply %cst_309, %451 : tensor<1x784x1024xf32>
    %453 = stablehlo.tanh %452 : tensor<1x784x1024xf32>
    %454 = stablehlo.add %cst_310, %453 : tensor<1x784x1024xf32>
    %455 = stablehlo.multiply %cst_311, %454 : tensor<1x784x1024xf32>
    %456 = stablehlo.multiply %447, %455 : tensor<1x784x1024xf32>
    %457 = stablehlo.dot_general %456, %cst_307, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x784x1024xf32>, tensor<1024x256xf32>) -> tensor<1x784x256xf32>
    %458 = stablehlo.reshape %cst_306 : (tensor<1x1x256xf32>) -> tensor<1x256xf32>
    %459 = stablehlo.broadcast_in_dim %458, dims = [0, 2] : (tensor<1x256xf32>) -> tensor<1x784x256xf32>
    %460 = stablehlo.add %457, %459 : tensor<1x784x256xf32>
    %461 = stablehlo.add %419, %460 : tensor<1x784x256xf32>
    %462 = stablehlo.reduce(%461 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x784x256xf32>, tensor<f32>) -> tensor<1x784xf32>
    %463 = stablehlo.divide %462, %cst_324 : tensor<1x784xf32>
    %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x1xf32>
    %465 = stablehlo.reshape %464 : (tensor<1x784x1xf32>) -> tensor<1x784xf32>
    %466 = stablehlo.broadcast_in_dim %465, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x256xf32>
    %467 = stablehlo.subtract %461, %466 : tensor<1x784x256xf32>
    %468 = stablehlo.multiply %461, %461 : tensor<1x784x256xf32>
    %469 = stablehlo.reduce(%468 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x784x256xf32>, tensor<f32>) -> tensor<1x784xf32>
    %470 = stablehlo.divide %469, %cst_324 : tensor<1x784xf32>
    %471 = stablehlo.multiply %463, %463 : tensor<1x784xf32>
    %472 = stablehlo.subtract %470, %471 : tensor<1x784xf32>
    %473 = stablehlo.maximum %cst_329, %472 : tensor<1x784xf32>
    %474 = stablehlo.broadcast_in_dim %473, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x1xf32>
    %475 = stablehlo.add %474, %cst_328 : tensor<1x784x1xf32>
    %476 = stablehlo.rsqrt %475 : tensor<1x784x1xf32>
    %477 = stablehlo.reshape %476 : (tensor<1x784x1xf32>) -> tensor<1x784xf32>
    %478 = stablehlo.broadcast_in_dim %477, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x256xf32>
    %479 = stablehlo.reshape %cst_305 : (tensor<1x1x256xf32>) -> tensor<1x256xf32>
    %480 = stablehlo.broadcast_in_dim %479, dims = [0, 2] : (tensor<1x256xf32>) -> tensor<1x784x256xf32>
    %481 = stablehlo.multiply %478, %480 : tensor<1x784x256xf32>
    %482 = stablehlo.multiply %467, %481 : tensor<1x784x256xf32>
    %483 = stablehlo.reshape %cst_304 : (tensor<1x1x256xf32>) -> tensor<1x256xf32>
    %484 = stablehlo.broadcast_in_dim %483, dims = [0, 2] : (tensor<1x256xf32>) -> tensor<1x784x256xf32>
    %485 = stablehlo.add %482, %484 : tensor<1x784x256xf32>
    %486 = stablehlo.reshape %485 : (tensor<1x784x256xf32>) -> tensor<1x28x28x256xf32>
    %487 = stablehlo.slice %486 [0:1, 3:28, 0:28, 0:256] : (tensor<1x28x28x256xf32>) -> tensor<1x25x28x256xf32>
    %488 = stablehlo.slice %486 [0:1, 0:3, 0:28, 0:256] : (tensor<1x28x28x256xf32>) -> tensor<1x3x28x256xf32>
    %489 = stablehlo.concatenate %487, %488, dim = 1 : (tensor<1x25x28x256xf32>, tensor<1x3x28x256xf32>) -> tensor<1x28x28x256xf32>
    %490 = stablehlo.slice %489 [0:1, 0:28, 3:28, 0:256] : (tensor<1x28x28x256xf32>) -> tensor<1x28x25x256xf32>
    %491 = stablehlo.slice %489 [0:1, 0:28, 0:3, 0:256] : (tensor<1x28x28x256xf32>) -> tensor<1x28x3x256xf32>
    %492 = stablehlo.concatenate %490, %491, dim = 2 : (tensor<1x28x25x256xf32>, tensor<1x28x3x256xf32>) -> tensor<1x28x28x256xf32>
    %493 = stablehlo.reshape %492 : (tensor<1x28x28x256xf32>) -> tensor<1x4x7x4x7x256xf32>
    %494 = stablehlo.transpose %493, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,4,4,7,7,256]{5,4,2,3,1,0}"} : (tensor<1x4x7x4x7x256xf32>) -> tensor<1x4x4x7x7x256xf32>
    %495 = stablehlo.reshape %494 : (tensor<1x4x4x7x7x256xf32>) -> tensor<16x49x256xf32>
    %496 = stablehlo.dot_general %495, %cst_303, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x49x256xf32>, tensor<256x768xf32>) -> tensor<16x49x768xf32>
    %497 = stablehlo.reshape %cst_302 : (tensor<1x1x768xf32>) -> tensor<768xf32>
    %498 = stablehlo.broadcast_in_dim %497, dims = [2] : (tensor<768xf32>) -> tensor<16x49x768xf32>
    %499 = stablehlo.add %496, %498 : tensor<16x49x768xf32>
    %500 = stablehlo.reshape %499 : (tensor<16x49x768xf32>) -> tensor<16x49x3x8x32xf32>
    %501 = stablehlo.transpose %500, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,16,8,49,32]{4,2,0,3,1}"} : (tensor<16x49x3x8x32xf32>) -> tensor<3x16x8x49x32xf32>
    %502 = stablehlo.slice %501 [0:1, 0:16, 0:8, 0:49, 0:32] : (tensor<3x16x8x49x32xf32>) -> tensor<1x16x8x49x32xf32>
    %503 = stablehlo.reshape %502 : (tensor<1x16x8x49x32xf32>) -> tensor<16x8x49x32xf32>
    %504 = stablehlo.multiply %503, %cst_319 : tensor<16x8x49x32xf32>
    %505 = stablehlo.slice %501 [1:2, 0:16, 0:8, 0:49, 0:32] : (tensor<3x16x8x49x32xf32>) -> tensor<1x16x8x49x32xf32>
    %506 = stablehlo.reshape %505 : (tensor<1x16x8x49x32xf32>) -> tensor<16x8x49x32xf32>
    %507 = stablehlo.transpose %506, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[16,8,32,49]{2,3,1,0}"} : (tensor<16x8x49x32xf32>) -> tensor<16x8x32x49xf32>
    %508 = stablehlo.dot_general %504, %507, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<16x8x49x32xf32>, tensor<16x8x32x49xf32>) -> tensor<16x8x49x49xf32>
    %509 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %510 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %511 = stablehlo.select %509, %510, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %512 = stablehlo.broadcast_in_dim %511, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %513 = "stablehlo.gather"(%cst_301, %512) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 8>}> : (tensor<169x8xf32>, tensor<2401x1xi32>) -> tensor<2401x8xf32>
    %514 = stablehlo.reshape %513 : (tensor<2401x8xf32>) -> tensor<49x49x8xf32>
    %515 = stablehlo.transpose %514, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[8,49,49]{0,2,1}"} : (tensor<49x49x8xf32>) -> tensor<8x49x49xf32>
    %516 = stablehlo.broadcast_in_dim %515, dims = [1, 2, 3] : (tensor<8x49x49xf32>) -> tensor<1x8x49x49xf32>
    %517 = stablehlo.reshape %516 : (tensor<1x8x49x49xf32>) -> tensor<8x49x49xf32>
    %518 = stablehlo.broadcast_in_dim %517, dims = [1, 2, 3] : (tensor<8x49x49xf32>) -> tensor<16x8x49x49xf32>
    %519 = stablehlo.add %508, %518 : tensor<16x8x49x49xf32>
    %520 = stablehlo.reduce(%519 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<16x8x49x49xf32>, tensor<f32>) -> tensor<16x8x49xf32>
    %521 = stablehlo.broadcast_in_dim %520, dims = [0, 1, 2] : (tensor<16x8x49xf32>) -> tensor<16x8x49x1xf32>
    %522 = stablehlo.reshape %521 : (tensor<16x8x49x1xf32>) -> tensor<16x8x49xf32>
    %523 = stablehlo.broadcast_in_dim %522, dims = [0, 1, 2] : (tensor<16x8x49xf32>) -> tensor<16x8x49x49xf32>
    %524 = stablehlo.subtract %519, %523 : tensor<16x8x49x49xf32>
    %525 = stablehlo.exponential %524 : tensor<16x8x49x49xf32>
    %526 = stablehlo.reduce(%525 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<16x8x49x49xf32>, tensor<f32>) -> tensor<16x8x49xf32>
    %527 = stablehlo.broadcast_in_dim %526, dims = [0, 1, 2] : (tensor<16x8x49xf32>) -> tensor<16x8x49x1xf32>
    %528 = stablehlo.reshape %527 : (tensor<16x8x49x1xf32>) -> tensor<16x8x49xf32>
    %529 = stablehlo.broadcast_in_dim %528, dims = [0, 1, 2] : (tensor<16x8x49xf32>) -> tensor<16x8x49x49xf32>
    %530 = stablehlo.divide %525, %529 : tensor<16x8x49x49xf32>
    %531 = stablehlo.slice %501 [2:3, 0:16, 0:8, 0:49, 0:32] : (tensor<3x16x8x49x32xf32>) -> tensor<1x16x8x49x32xf32>
    %532 = stablehlo.reshape %531 : (tensor<1x16x8x49x32xf32>) -> tensor<16x8x49x32xf32>
    %533 = stablehlo.dot_general %530, %532, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<16x8x49x49xf32>, tensor<16x8x49x32xf32>) -> tensor<16x8x49x32xf32>
    %534 = stablehlo.transpose %533, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,49,8,32]{3,1,2,0}"} : (tensor<16x8x49x32xf32>) -> tensor<16x49x8x32xf32>
    %535 = stablehlo.reshape %534 : (tensor<16x49x8x32xf32>) -> tensor<16x49x256xf32>
    %536 = stablehlo.dot_general %535, %cst_300, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x49x256xf32>, tensor<256x256xf32>) -> tensor<16x49x256xf32>
    %537 = stablehlo.reshape %cst_299 : (tensor<1x1x256xf32>) -> tensor<256xf32>
    %538 = stablehlo.broadcast_in_dim %537, dims = [2] : (tensor<256xf32>) -> tensor<16x49x256xf32>
    %539 = stablehlo.add %536, %538 : tensor<16x49x256xf32>
    %540 = stablehlo.reshape %539 : (tensor<16x49x256xf32>) -> tensor<1x4x4x7x7x256xf32>
    %541 = stablehlo.transpose %540, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,4,7,4,7,256]{5,4,2,3,1,0}"} : (tensor<1x4x4x7x7x256xf32>) -> tensor<1x4x7x4x7x256xf32>
    %542 = stablehlo.reshape %541 : (tensor<1x4x7x4x7x256xf32>) -> tensor<1x28x28x256xf32>
    %543 = stablehlo.slice %542 [0:1, 25:28, 0:28, 0:256] : (tensor<1x28x28x256xf32>) -> tensor<1x3x28x256xf32>
    %544 = stablehlo.slice %542 [0:1, 0:25, 0:28, 0:256] : (tensor<1x28x28x256xf32>) -> tensor<1x25x28x256xf32>
    %545 = stablehlo.concatenate %543, %544, dim = 1 : (tensor<1x3x28x256xf32>, tensor<1x25x28x256xf32>) -> tensor<1x28x28x256xf32>
    %546 = stablehlo.slice %545 [0:1, 0:28, 25:28, 0:256] : (tensor<1x28x28x256xf32>) -> tensor<1x28x3x256xf32>
    %547 = stablehlo.slice %545 [0:1, 0:28, 0:25, 0:256] : (tensor<1x28x28x256xf32>) -> tensor<1x28x25x256xf32>
    %548 = stablehlo.concatenate %546, %547, dim = 2 : (tensor<1x28x3x256xf32>, tensor<1x28x25x256xf32>) -> tensor<1x28x28x256xf32>
    %549 = stablehlo.reshape %548 : (tensor<1x28x28x256xf32>) -> tensor<1x784x256xf32>
    %550 = stablehlo.add %461, %549 : tensor<1x784x256xf32>
    %551 = stablehlo.reduce(%550 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x784x256xf32>, tensor<f32>) -> tensor<1x784xf32>
    %552 = stablehlo.divide %551, %cst_324 : tensor<1x784xf32>
    %553 = stablehlo.broadcast_in_dim %552, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x1xf32>
    %554 = stablehlo.reshape %553 : (tensor<1x784x1xf32>) -> tensor<1x784xf32>
    %555 = stablehlo.broadcast_in_dim %554, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x256xf32>
    %556 = stablehlo.subtract %550, %555 : tensor<1x784x256xf32>
    %557 = stablehlo.multiply %550, %550 : tensor<1x784x256xf32>
    %558 = stablehlo.reduce(%557 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x784x256xf32>, tensor<f32>) -> tensor<1x784xf32>
    %559 = stablehlo.divide %558, %cst_324 : tensor<1x784xf32>
    %560 = stablehlo.multiply %552, %552 : tensor<1x784xf32>
    %561 = stablehlo.subtract %559, %560 : tensor<1x784xf32>
    %562 = stablehlo.maximum %cst_329, %561 : tensor<1x784xf32>
    %563 = stablehlo.broadcast_in_dim %562, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x1xf32>
    %564 = stablehlo.add %563, %cst_328 : tensor<1x784x1xf32>
    %565 = stablehlo.rsqrt %564 : tensor<1x784x1xf32>
    %566 = stablehlo.reshape %565 : (tensor<1x784x1xf32>) -> tensor<1x784xf32>
    %567 = stablehlo.broadcast_in_dim %566, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<1x784x256xf32>
    %568 = stablehlo.reshape %cst_298 : (tensor<1x1x256xf32>) -> tensor<1x256xf32>
    %569 = stablehlo.broadcast_in_dim %568, dims = [0, 2] : (tensor<1x256xf32>) -> tensor<1x784x256xf32>
    %570 = stablehlo.multiply %567, %569 : tensor<1x784x256xf32>
    %571 = stablehlo.multiply %556, %570 : tensor<1x784x256xf32>
    %572 = stablehlo.reshape %cst_297 : (tensor<1x1x256xf32>) -> tensor<1x256xf32>
    %573 = stablehlo.broadcast_in_dim %572, dims = [0, 2] : (tensor<1x256xf32>) -> tensor<1x784x256xf32>
    %574 = stablehlo.add %571, %573 : tensor<1x784x256xf32>
    %575 = stablehlo.dot_general %574, %cst_296, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x784x256xf32>, tensor<256x1024xf32>) -> tensor<1x784x1024xf32>
    %576 = stablehlo.reshape %cst_295 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %577 = stablehlo.broadcast_in_dim %576, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x784x1024xf32>
    %578 = stablehlo.add %575, %577 : tensor<1x784x1024xf32>
    %579 = stablehlo.multiply %578, %578 : tensor<1x784x1024xf32>
    %580 = stablehlo.multiply %579, %578 : tensor<1x784x1024xf32>
    %581 = stablehlo.multiply %cst_308, %580 : tensor<1x784x1024xf32>
    %582 = stablehlo.add %578, %581 : tensor<1x784x1024xf32>
    %583 = stablehlo.multiply %cst_309, %582 : tensor<1x784x1024xf32>
    %584 = stablehlo.tanh %583 : tensor<1x784x1024xf32>
    %585 = stablehlo.add %cst_310, %584 : tensor<1x784x1024xf32>
    %586 = stablehlo.multiply %cst_311, %585 : tensor<1x784x1024xf32>
    %587 = stablehlo.multiply %578, %586 : tensor<1x784x1024xf32>
    %588 = stablehlo.dot_general %587, %cst_294, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x784x1024xf32>, tensor<1024x256xf32>) -> tensor<1x784x256xf32>
    %589 = stablehlo.reshape %cst_293 : (tensor<1x1x256xf32>) -> tensor<1x256xf32>
    %590 = stablehlo.broadcast_in_dim %589, dims = [0, 2] : (tensor<1x256xf32>) -> tensor<1x784x256xf32>
    %591 = stablehlo.add %588, %590 : tensor<1x784x256xf32>
    %592 = stablehlo.add %550, %591 : tensor<1x784x256xf32>
    %593 = stablehlo.reshape %592 : (tensor<1x784x256xf32>) -> tensor<1x28x28x256xf32>
    %594 = stablehlo.iota dim = 0 : tensor<14xi32>
    %595 = stablehlo.multiply %c_291, %594 : tensor<14xi32>
    %596 = stablehlo.add %c_292, %595 : tensor<14xi32>
    %597 = stablehlo.broadcast_in_dim %596, dims = [0] : (tensor<14xi32>) -> tensor<14x14x1xi32>
    %598 = stablehlo.iota dim = 0 : tensor<14xi32>
    %599 = stablehlo.multiply %c_291, %598 : tensor<14xi32>
    %600 = stablehlo.add %c_292, %599 : tensor<14xi32>
    %601 = stablehlo.broadcast_in_dim %600, dims = [1] : (tensor<14xi32>) -> tensor<14x14x1xi32>
    %602 = stablehlo.concatenate %597, %601, dim = 2 : (tensor<14x14x1xi32>, tensor<14x14x1xi32>) -> tensor<14x14x2xi32>
    %603 = "stablehlo.gather"(%593, %602) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 3], collapsed_slice_dims = [1, 2], start_index_map = [1, 2], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 1, 256>}> : (tensor<1x28x28x256xf32>, tensor<14x14x2xi32>) -> tensor<1x14x14x256xf32>
    %604 = stablehlo.iota dim = 0 : tensor<14xi32>
    %605 = stablehlo.multiply %c_291, %604 : tensor<14xi32>
    %606 = stablehlo.add %c_290, %605 : tensor<14xi32>
    %607 = stablehlo.broadcast_in_dim %606, dims = [0] : (tensor<14xi32>) -> tensor<14x14x1xi32>
    %608 = stablehlo.iota dim = 0 : tensor<14xi32>
    %609 = stablehlo.multiply %c_291, %608 : tensor<14xi32>
    %610 = stablehlo.add %c_292, %609 : tensor<14xi32>
    %611 = stablehlo.broadcast_in_dim %610, dims = [1] : (tensor<14xi32>) -> tensor<14x14x1xi32>
    %612 = stablehlo.concatenate %607, %611, dim = 2 : (tensor<14x14x1xi32>, tensor<14x14x1xi32>) -> tensor<14x14x2xi32>
    %613 = "stablehlo.gather"(%593, %612) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 3], collapsed_slice_dims = [1, 2], start_index_map = [1, 2], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 1, 256>}> : (tensor<1x28x28x256xf32>, tensor<14x14x2xi32>) -> tensor<1x14x14x256xf32>
    %614 = stablehlo.iota dim = 0 : tensor<14xi32>
    %615 = stablehlo.multiply %c_291, %614 : tensor<14xi32>
    %616 = stablehlo.add %c_292, %615 : tensor<14xi32>
    %617 = stablehlo.broadcast_in_dim %616, dims = [0] : (tensor<14xi32>) -> tensor<14x14x1xi32>
    %618 = stablehlo.iota dim = 0 : tensor<14xi32>
    %619 = stablehlo.multiply %c_291, %618 : tensor<14xi32>
    %620 = stablehlo.add %c_290, %619 : tensor<14xi32>
    %621 = stablehlo.broadcast_in_dim %620, dims = [1] : (tensor<14xi32>) -> tensor<14x14x1xi32>
    %622 = stablehlo.concatenate %617, %621, dim = 2 : (tensor<14x14x1xi32>, tensor<14x14x1xi32>) -> tensor<14x14x2xi32>
    %623 = "stablehlo.gather"(%593, %622) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 3], collapsed_slice_dims = [1, 2], start_index_map = [1, 2], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 1, 256>}> : (tensor<1x28x28x256xf32>, tensor<14x14x2xi32>) -> tensor<1x14x14x256xf32>
    %624 = stablehlo.iota dim = 0 : tensor<14xi32>
    %625 = stablehlo.multiply %c_291, %624 : tensor<14xi32>
    %626 = stablehlo.add %c_290, %625 : tensor<14xi32>
    %627 = stablehlo.broadcast_in_dim %626, dims = [0] : (tensor<14xi32>) -> tensor<14x14x1xi32>
    %628 = stablehlo.iota dim = 0 : tensor<14xi32>
    %629 = stablehlo.multiply %c_291, %628 : tensor<14xi32>
    %630 = stablehlo.add %c_290, %629 : tensor<14xi32>
    %631 = stablehlo.broadcast_in_dim %630, dims = [1] : (tensor<14xi32>) -> tensor<14x14x1xi32>
    %632 = stablehlo.concatenate %627, %631, dim = 2 : (tensor<14x14x1xi32>, tensor<14x14x1xi32>) -> tensor<14x14x2xi32>
    %633 = "stablehlo.gather"(%593, %632) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 3], collapsed_slice_dims = [1, 2], start_index_map = [1, 2], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 1, 256>}> : (tensor<1x28x28x256xf32>, tensor<14x14x2xi32>) -> tensor<1x14x14x256xf32>
    %634 = stablehlo.concatenate %603, %613, %623, %633, dim = 3 : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x1024xf32>
    %635 = stablehlo.reshape %634 : (tensor<1x14x14x1024xf32>) -> tensor<1x196x1024xf32>
    %636 = stablehlo.reduce(%635 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x1024xf32>, tensor<f32>) -> tensor<1x196xf32>
    %637 = stablehlo.divide %636, %cst_289 : tensor<1x196xf32>
    %638 = stablehlo.broadcast_in_dim %637, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %639 = stablehlo.reshape %638 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %640 = stablehlo.broadcast_in_dim %639, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1024xf32>
    %641 = stablehlo.subtract %635, %640 : tensor<1x196x1024xf32>
    %642 = stablehlo.multiply %635, %635 : tensor<1x196x1024xf32>
    %643 = stablehlo.reduce(%642 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x1024xf32>, tensor<f32>) -> tensor<1x196xf32>
    %644 = stablehlo.divide %643, %cst_289 : tensor<1x196xf32>
    %645 = stablehlo.multiply %637, %637 : tensor<1x196xf32>
    %646 = stablehlo.subtract %644, %645 : tensor<1x196xf32>
    %647 = stablehlo.maximum %cst_288, %646 : tensor<1x196xf32>
    %648 = stablehlo.broadcast_in_dim %647, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %649 = stablehlo.add %648, %cst_287 : tensor<1x196x1xf32>
    %650 = stablehlo.rsqrt %649 : tensor<1x196x1xf32>
    %651 = stablehlo.reshape %650 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %652 = stablehlo.broadcast_in_dim %651, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1024xf32>
    %653 = stablehlo.reshape %cst_286 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %654 = stablehlo.broadcast_in_dim %653, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x196x1024xf32>
    %655 = stablehlo.multiply %652, %654 : tensor<1x196x1024xf32>
    %656 = stablehlo.multiply %641, %655 : tensor<1x196x1024xf32>
    %657 = stablehlo.reshape %cst_285 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %658 = stablehlo.broadcast_in_dim %657, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x196x1024xf32>
    %659 = stablehlo.add %656, %658 : tensor<1x196x1024xf32>
    %660 = stablehlo.dot_general %659, %cst_284, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x1024xf32>, tensor<1024x512xf32>) -> tensor<1x196x512xf32>
    %661 = stablehlo.reduce(%660 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %662 = stablehlo.divide %661, %cst_283 : tensor<1x196xf32>
    %663 = stablehlo.broadcast_in_dim %662, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %664 = stablehlo.reshape %663 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %665 = stablehlo.broadcast_in_dim %664, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %666 = stablehlo.subtract %660, %665 : tensor<1x196x512xf32>
    %667 = stablehlo.multiply %660, %660 : tensor<1x196x512xf32>
    %668 = stablehlo.reduce(%667 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %669 = stablehlo.divide %668, %cst_283 : tensor<1x196xf32>
    %670 = stablehlo.multiply %662, %662 : tensor<1x196xf32>
    %671 = stablehlo.subtract %669, %670 : tensor<1x196xf32>
    %672 = stablehlo.maximum %cst_288, %671 : tensor<1x196xf32>
    %673 = stablehlo.broadcast_in_dim %672, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %674 = stablehlo.add %673, %cst_287 : tensor<1x196x1xf32>
    %675 = stablehlo.rsqrt %674 : tensor<1x196x1xf32>
    %676 = stablehlo.reshape %675 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %677 = stablehlo.broadcast_in_dim %676, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %678 = stablehlo.reshape %cst_282 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %679 = stablehlo.broadcast_in_dim %678, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %680 = stablehlo.multiply %677, %679 : tensor<1x196x512xf32>
    %681 = stablehlo.multiply %666, %680 : tensor<1x196x512xf32>
    %682 = stablehlo.reshape %cst_281 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %683 = stablehlo.broadcast_in_dim %682, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %684 = stablehlo.add %681, %683 : tensor<1x196x512xf32>
    %685 = stablehlo.reshape %684 : (tensor<1x196x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %686 = stablehlo.transpose %685, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %687 = stablehlo.reshape %686 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %688 = stablehlo.dot_general %687, %cst_280, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %689 = stablehlo.reshape %cst_279 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %690 = stablehlo.broadcast_in_dim %689, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %691 = stablehlo.add %688, %690 : tensor<4x49x1536xf32>
    %692 = stablehlo.reshape %691 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %693 = stablehlo.transpose %692, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %694 = stablehlo.slice %693 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %695 = stablehlo.reshape %694 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %696 = stablehlo.multiply %695, %cst_278 : tensor<4x16x49x32xf32>
    %697 = stablehlo.slice %693 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %698 = stablehlo.reshape %697 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %699 = stablehlo.transpose %698, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %700 = stablehlo.dot_general %696, %699, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %701 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %702 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %703 = stablehlo.select %701, %702, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %704 = stablehlo.broadcast_in_dim %703, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %705 = "stablehlo.gather"(%cst_277, %704) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %706 = stablehlo.reshape %705 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %707 = stablehlo.transpose %706, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %708 = stablehlo.broadcast_in_dim %707, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %709 = stablehlo.reshape %708 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %710 = stablehlo.broadcast_in_dim %709, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %711 = stablehlo.add %700, %710 : tensor<4x16x49x49xf32>
    %712 = stablehlo.reduce(%711 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %713 = stablehlo.broadcast_in_dim %712, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %714 = stablehlo.reshape %713 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %715 = stablehlo.broadcast_in_dim %714, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %716 = stablehlo.subtract %711, %715 : tensor<4x16x49x49xf32>
    %717 = stablehlo.exponential %716 : tensor<4x16x49x49xf32>
    %718 = stablehlo.reduce(%717 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %719 = stablehlo.broadcast_in_dim %718, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %720 = stablehlo.reshape %719 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %721 = stablehlo.broadcast_in_dim %720, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %722 = stablehlo.divide %717, %721 : tensor<4x16x49x49xf32>
    %723 = stablehlo.slice %693 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %724 = stablehlo.reshape %723 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %725 = stablehlo.dot_general %722, %724, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %726 = stablehlo.transpose %725, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %727 = stablehlo.reshape %726 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %728 = stablehlo.dot_general %727, %cst_276, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %729 = stablehlo.reshape %cst_275 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %730 = stablehlo.broadcast_in_dim %729, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %731 = stablehlo.add %728, %730 : tensor<4x49x512xf32>
    %732 = stablehlo.reshape %731 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %733 = stablehlo.transpose %732, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %734 = stablehlo.reshape %733 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x196x512xf32>
    %735 = stablehlo.add %660, %734 : tensor<1x196x512xf32>
    %736 = stablehlo.reduce(%735 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %737 = stablehlo.divide %736, %cst_283 : tensor<1x196xf32>
    %738 = stablehlo.broadcast_in_dim %737, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %739 = stablehlo.reshape %738 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %740 = stablehlo.broadcast_in_dim %739, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %741 = stablehlo.subtract %735, %740 : tensor<1x196x512xf32>
    %742 = stablehlo.multiply %735, %735 : tensor<1x196x512xf32>
    %743 = stablehlo.reduce(%742 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %744 = stablehlo.divide %743, %cst_283 : tensor<1x196xf32>
    %745 = stablehlo.multiply %737, %737 : tensor<1x196xf32>
    %746 = stablehlo.subtract %744, %745 : tensor<1x196xf32>
    %747 = stablehlo.maximum %cst_288, %746 : tensor<1x196xf32>
    %748 = stablehlo.broadcast_in_dim %747, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %749 = stablehlo.add %748, %cst_287 : tensor<1x196x1xf32>
    %750 = stablehlo.rsqrt %749 : tensor<1x196x1xf32>
    %751 = stablehlo.reshape %750 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %752 = stablehlo.broadcast_in_dim %751, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %753 = stablehlo.reshape %cst_274 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %754 = stablehlo.broadcast_in_dim %753, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %755 = stablehlo.multiply %752, %754 : tensor<1x196x512xf32>
    %756 = stablehlo.multiply %741, %755 : tensor<1x196x512xf32>
    %757 = stablehlo.reshape %cst_273 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %758 = stablehlo.broadcast_in_dim %757, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %759 = stablehlo.add %756, %758 : tensor<1x196x512xf32>
    %760 = stablehlo.dot_general %759, %cst_272, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %761 = stablehlo.reshape %cst_271 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %762 = stablehlo.broadcast_in_dim %761, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %763 = stablehlo.add %760, %762 : tensor<1x196x2048xf32>
    %764 = stablehlo.multiply %763, %763 : tensor<1x196x2048xf32>
    %765 = stablehlo.multiply %764, %763 : tensor<1x196x2048xf32>
    %766 = stablehlo.multiply %cst_267, %765 : tensor<1x196x2048xf32>
    %767 = stablehlo.add %763, %766 : tensor<1x196x2048xf32>
    %768 = stablehlo.multiply %cst_268, %767 : tensor<1x196x2048xf32>
    %769 = stablehlo.tanh %768 : tensor<1x196x2048xf32>
    %770 = stablehlo.add %cst_269, %769 : tensor<1x196x2048xf32>
    %771 = stablehlo.multiply %cst_270, %770 : tensor<1x196x2048xf32>
    %772 = stablehlo.multiply %763, %771 : tensor<1x196x2048xf32>
    %773 = stablehlo.dot_general %772, %cst_266, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %774 = stablehlo.reshape %cst_265 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %775 = stablehlo.broadcast_in_dim %774, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %776 = stablehlo.add %773, %775 : tensor<1x196x512xf32>
    %777 = stablehlo.add %735, %776 : tensor<1x196x512xf32>
    %778 = stablehlo.reduce(%777 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %779 = stablehlo.divide %778, %cst_283 : tensor<1x196xf32>
    %780 = stablehlo.broadcast_in_dim %779, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %781 = stablehlo.reshape %780 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %782 = stablehlo.broadcast_in_dim %781, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %783 = stablehlo.subtract %777, %782 : tensor<1x196x512xf32>
    %784 = stablehlo.multiply %777, %777 : tensor<1x196x512xf32>
    %785 = stablehlo.reduce(%784 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %786 = stablehlo.divide %785, %cst_283 : tensor<1x196xf32>
    %787 = stablehlo.multiply %779, %779 : tensor<1x196xf32>
    %788 = stablehlo.subtract %786, %787 : tensor<1x196xf32>
    %789 = stablehlo.maximum %cst_288, %788 : tensor<1x196xf32>
    %790 = stablehlo.broadcast_in_dim %789, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %791 = stablehlo.add %790, %cst_287 : tensor<1x196x1xf32>
    %792 = stablehlo.rsqrt %791 : tensor<1x196x1xf32>
    %793 = stablehlo.reshape %792 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %794 = stablehlo.broadcast_in_dim %793, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %795 = stablehlo.reshape %cst_264 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %796 = stablehlo.broadcast_in_dim %795, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %797 = stablehlo.multiply %794, %796 : tensor<1x196x512xf32>
    %798 = stablehlo.multiply %783, %797 : tensor<1x196x512xf32>
    %799 = stablehlo.reshape %cst_263 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %800 = stablehlo.broadcast_in_dim %799, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %801 = stablehlo.add %798, %800 : tensor<1x196x512xf32>
    %802 = stablehlo.reshape %801 : (tensor<1x196x512xf32>) -> tensor<1x14x14x512xf32>
    %803 = stablehlo.slice %802 [0:1, 3:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %804 = stablehlo.slice %802 [0:1, 0:3, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %805 = stablehlo.concatenate %803, %804, dim = 1 : (tensor<1x11x14x512xf32>, tensor<1x3x14x512xf32>) -> tensor<1x14x14x512xf32>
    %806 = stablehlo.slice %805 [0:1, 0:14, 3:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %807 = stablehlo.slice %805 [0:1, 0:14, 0:3, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %808 = stablehlo.concatenate %806, %807, dim = 2 : (tensor<1x14x11x512xf32>, tensor<1x14x3x512xf32>) -> tensor<1x14x14x512xf32>
    %809 = stablehlo.reshape %808 : (tensor<1x14x14x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %810 = stablehlo.transpose %809, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %811 = stablehlo.reshape %810 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %812 = stablehlo.dot_general %811, %cst_262, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %813 = stablehlo.reshape %cst_261 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %814 = stablehlo.broadcast_in_dim %813, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %815 = stablehlo.add %812, %814 : tensor<4x49x1536xf32>
    %816 = stablehlo.reshape %815 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %817 = stablehlo.transpose %816, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %818 = stablehlo.slice %817 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %819 = stablehlo.reshape %818 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %820 = stablehlo.multiply %819, %cst_278 : tensor<4x16x49x32xf32>
    %821 = stablehlo.slice %817 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %822 = stablehlo.reshape %821 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %823 = stablehlo.transpose %822, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %824 = stablehlo.dot_general %820, %823, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %825 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %826 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %827 = stablehlo.select %825, %826, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %828 = stablehlo.broadcast_in_dim %827, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %829 = "stablehlo.gather"(%cst_260, %828) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %830 = stablehlo.reshape %829 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %831 = stablehlo.transpose %830, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %832 = stablehlo.broadcast_in_dim %831, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %833 = stablehlo.reshape %832 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %834 = stablehlo.broadcast_in_dim %833, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %835 = stablehlo.add %824, %834 : tensor<4x16x49x49xf32>
    %836 = stablehlo.reduce(%835 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %837 = stablehlo.broadcast_in_dim %836, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %838 = stablehlo.reshape %837 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %839 = stablehlo.broadcast_in_dim %838, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %840 = stablehlo.subtract %835, %839 : tensor<4x16x49x49xf32>
    %841 = stablehlo.exponential %840 : tensor<4x16x49x49xf32>
    %842 = stablehlo.reduce(%841 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %843 = stablehlo.broadcast_in_dim %842, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %844 = stablehlo.reshape %843 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %845 = stablehlo.broadcast_in_dim %844, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %846 = stablehlo.divide %841, %845 : tensor<4x16x49x49xf32>
    %847 = stablehlo.slice %817 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %848 = stablehlo.reshape %847 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %849 = stablehlo.dot_general %846, %848, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %850 = stablehlo.transpose %849, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %851 = stablehlo.reshape %850 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %852 = stablehlo.dot_general %851, %cst_259, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %853 = stablehlo.reshape %cst_258 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %854 = stablehlo.broadcast_in_dim %853, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %855 = stablehlo.add %852, %854 : tensor<4x49x512xf32>
    %856 = stablehlo.reshape %855 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %857 = stablehlo.transpose %856, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %858 = stablehlo.reshape %857 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x14x14x512xf32>
    %859 = stablehlo.slice %858 [0:1, 11:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %860 = stablehlo.slice %858 [0:1, 0:11, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %861 = stablehlo.concatenate %859, %860, dim = 1 : (tensor<1x3x14x512xf32>, tensor<1x11x14x512xf32>) -> tensor<1x14x14x512xf32>
    %862 = stablehlo.slice %861 [0:1, 0:14, 11:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %863 = stablehlo.slice %861 [0:1, 0:14, 0:11, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %864 = stablehlo.concatenate %862, %863, dim = 2 : (tensor<1x14x3x512xf32>, tensor<1x14x11x512xf32>) -> tensor<1x14x14x512xf32>
    %865 = stablehlo.reshape %864 : (tensor<1x14x14x512xf32>) -> tensor<1x196x512xf32>
    %866 = stablehlo.add %777, %865 : tensor<1x196x512xf32>
    %867 = stablehlo.reduce(%866 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %868 = stablehlo.divide %867, %cst_283 : tensor<1x196xf32>
    %869 = stablehlo.broadcast_in_dim %868, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %870 = stablehlo.reshape %869 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %871 = stablehlo.broadcast_in_dim %870, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %872 = stablehlo.subtract %866, %871 : tensor<1x196x512xf32>
    %873 = stablehlo.multiply %866, %866 : tensor<1x196x512xf32>
    %874 = stablehlo.reduce(%873 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %875 = stablehlo.divide %874, %cst_283 : tensor<1x196xf32>
    %876 = stablehlo.multiply %868, %868 : tensor<1x196xf32>
    %877 = stablehlo.subtract %875, %876 : tensor<1x196xf32>
    %878 = stablehlo.maximum %cst_288, %877 : tensor<1x196xf32>
    %879 = stablehlo.broadcast_in_dim %878, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %880 = stablehlo.add %879, %cst_287 : tensor<1x196x1xf32>
    %881 = stablehlo.rsqrt %880 : tensor<1x196x1xf32>
    %882 = stablehlo.reshape %881 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %883 = stablehlo.broadcast_in_dim %882, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %884 = stablehlo.reshape %cst_257 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %885 = stablehlo.broadcast_in_dim %884, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %886 = stablehlo.multiply %883, %885 : tensor<1x196x512xf32>
    %887 = stablehlo.multiply %872, %886 : tensor<1x196x512xf32>
    %888 = stablehlo.reshape %cst_256 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %889 = stablehlo.broadcast_in_dim %888, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %890 = stablehlo.add %887, %889 : tensor<1x196x512xf32>
    %891 = stablehlo.dot_general %890, %cst_255, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %892 = stablehlo.reshape %cst_254 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %893 = stablehlo.broadcast_in_dim %892, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %894 = stablehlo.add %891, %893 : tensor<1x196x2048xf32>
    %895 = stablehlo.multiply %894, %894 : tensor<1x196x2048xf32>
    %896 = stablehlo.multiply %895, %894 : tensor<1x196x2048xf32>
    %897 = stablehlo.multiply %cst_267, %896 : tensor<1x196x2048xf32>
    %898 = stablehlo.add %894, %897 : tensor<1x196x2048xf32>
    %899 = stablehlo.multiply %cst_268, %898 : tensor<1x196x2048xf32>
    %900 = stablehlo.tanh %899 : tensor<1x196x2048xf32>
    %901 = stablehlo.add %cst_269, %900 : tensor<1x196x2048xf32>
    %902 = stablehlo.multiply %cst_270, %901 : tensor<1x196x2048xf32>
    %903 = stablehlo.multiply %894, %902 : tensor<1x196x2048xf32>
    %904 = stablehlo.dot_general %903, %cst_253, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %905 = stablehlo.reshape %cst_252 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %906 = stablehlo.broadcast_in_dim %905, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %907 = stablehlo.add %904, %906 : tensor<1x196x512xf32>
    %908 = stablehlo.add %866, %907 : tensor<1x196x512xf32>
    %909 = stablehlo.reduce(%908 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %910 = stablehlo.divide %909, %cst_283 : tensor<1x196xf32>
    %911 = stablehlo.broadcast_in_dim %910, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %912 = stablehlo.reshape %911 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %913 = stablehlo.broadcast_in_dim %912, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %914 = stablehlo.subtract %908, %913 : tensor<1x196x512xf32>
    %915 = stablehlo.multiply %908, %908 : tensor<1x196x512xf32>
    %916 = stablehlo.reduce(%915 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %917 = stablehlo.divide %916, %cst_283 : tensor<1x196xf32>
    %918 = stablehlo.multiply %910, %910 : tensor<1x196xf32>
    %919 = stablehlo.subtract %917, %918 : tensor<1x196xf32>
    %920 = stablehlo.maximum %cst_288, %919 : tensor<1x196xf32>
    %921 = stablehlo.broadcast_in_dim %920, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %922 = stablehlo.add %921, %cst_287 : tensor<1x196x1xf32>
    %923 = stablehlo.rsqrt %922 : tensor<1x196x1xf32>
    %924 = stablehlo.reshape %923 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %925 = stablehlo.broadcast_in_dim %924, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %926 = stablehlo.reshape %cst_251 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %927 = stablehlo.broadcast_in_dim %926, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %928 = stablehlo.multiply %925, %927 : tensor<1x196x512xf32>
    %929 = stablehlo.multiply %914, %928 : tensor<1x196x512xf32>
    %930 = stablehlo.reshape %cst_250 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %931 = stablehlo.broadcast_in_dim %930, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %932 = stablehlo.add %929, %931 : tensor<1x196x512xf32>
    %933 = stablehlo.reshape %932 : (tensor<1x196x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %934 = stablehlo.transpose %933, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %935 = stablehlo.reshape %934 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %936 = stablehlo.dot_general %935, %cst_249, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %937 = stablehlo.reshape %cst_248 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %938 = stablehlo.broadcast_in_dim %937, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %939 = stablehlo.add %936, %938 : tensor<4x49x1536xf32>
    %940 = stablehlo.reshape %939 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %941 = stablehlo.transpose %940, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %942 = stablehlo.slice %941 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %943 = stablehlo.reshape %942 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %944 = stablehlo.multiply %943, %cst_278 : tensor<4x16x49x32xf32>
    %945 = stablehlo.slice %941 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %946 = stablehlo.reshape %945 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %947 = stablehlo.transpose %946, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %948 = stablehlo.dot_general %944, %947, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %949 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %950 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %951 = stablehlo.select %949, %950, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %952 = stablehlo.broadcast_in_dim %951, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %953 = "stablehlo.gather"(%cst_247, %952) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %954 = stablehlo.reshape %953 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %955 = stablehlo.transpose %954, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %956 = stablehlo.broadcast_in_dim %955, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %957 = stablehlo.reshape %956 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %958 = stablehlo.broadcast_in_dim %957, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %959 = stablehlo.add %948, %958 : tensor<4x16x49x49xf32>
    %960 = stablehlo.reduce(%959 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %961 = stablehlo.broadcast_in_dim %960, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %962 = stablehlo.reshape %961 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %963 = stablehlo.broadcast_in_dim %962, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %964 = stablehlo.subtract %959, %963 : tensor<4x16x49x49xf32>
    %965 = stablehlo.exponential %964 : tensor<4x16x49x49xf32>
    %966 = stablehlo.reduce(%965 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %967 = stablehlo.broadcast_in_dim %966, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %968 = stablehlo.reshape %967 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %969 = stablehlo.broadcast_in_dim %968, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %970 = stablehlo.divide %965, %969 : tensor<4x16x49x49xf32>
    %971 = stablehlo.slice %941 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %972 = stablehlo.reshape %971 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %973 = stablehlo.dot_general %970, %972, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %974 = stablehlo.transpose %973, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %975 = stablehlo.reshape %974 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %976 = stablehlo.dot_general %975, %cst_246, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %977 = stablehlo.reshape %cst_245 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %978 = stablehlo.broadcast_in_dim %977, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %979 = stablehlo.add %976, %978 : tensor<4x49x512xf32>
    %980 = stablehlo.reshape %979 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %981 = stablehlo.transpose %980, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %982 = stablehlo.reshape %981 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x196x512xf32>
    %983 = stablehlo.add %908, %982 : tensor<1x196x512xf32>
    %984 = stablehlo.reduce(%983 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %985 = stablehlo.divide %984, %cst_283 : tensor<1x196xf32>
    %986 = stablehlo.broadcast_in_dim %985, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %987 = stablehlo.reshape %986 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %988 = stablehlo.broadcast_in_dim %987, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %989 = stablehlo.subtract %983, %988 : tensor<1x196x512xf32>
    %990 = stablehlo.multiply %983, %983 : tensor<1x196x512xf32>
    %991 = stablehlo.reduce(%990 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %992 = stablehlo.divide %991, %cst_283 : tensor<1x196xf32>
    %993 = stablehlo.multiply %985, %985 : tensor<1x196xf32>
    %994 = stablehlo.subtract %992, %993 : tensor<1x196xf32>
    %995 = stablehlo.maximum %cst_288, %994 : tensor<1x196xf32>
    %996 = stablehlo.broadcast_in_dim %995, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %997 = stablehlo.add %996, %cst_287 : tensor<1x196x1xf32>
    %998 = stablehlo.rsqrt %997 : tensor<1x196x1xf32>
    %999 = stablehlo.reshape %998 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1000 = stablehlo.broadcast_in_dim %999, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1001 = stablehlo.reshape %cst_244 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1002 = stablehlo.broadcast_in_dim %1001, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1003 = stablehlo.multiply %1000, %1002 : tensor<1x196x512xf32>
    %1004 = stablehlo.multiply %989, %1003 : tensor<1x196x512xf32>
    %1005 = stablehlo.reshape %cst_243 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1006 = stablehlo.broadcast_in_dim %1005, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1007 = stablehlo.add %1004, %1006 : tensor<1x196x512xf32>
    %1008 = stablehlo.dot_general %1007, %cst_242, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %1009 = stablehlo.reshape %cst_241 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %1010 = stablehlo.broadcast_in_dim %1009, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %1011 = stablehlo.add %1008, %1010 : tensor<1x196x2048xf32>
    %1012 = stablehlo.multiply %1011, %1011 : tensor<1x196x2048xf32>
    %1013 = stablehlo.multiply %1012, %1011 : tensor<1x196x2048xf32>
    %1014 = stablehlo.multiply %cst_267, %1013 : tensor<1x196x2048xf32>
    %1015 = stablehlo.add %1011, %1014 : tensor<1x196x2048xf32>
    %1016 = stablehlo.multiply %cst_268, %1015 : tensor<1x196x2048xf32>
    %1017 = stablehlo.tanh %1016 : tensor<1x196x2048xf32>
    %1018 = stablehlo.add %cst_269, %1017 : tensor<1x196x2048xf32>
    %1019 = stablehlo.multiply %cst_270, %1018 : tensor<1x196x2048xf32>
    %1020 = stablehlo.multiply %1011, %1019 : tensor<1x196x2048xf32>
    %1021 = stablehlo.dot_general %1020, %cst_240, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %1022 = stablehlo.reshape %cst_239 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1023 = stablehlo.broadcast_in_dim %1022, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1024 = stablehlo.add %1021, %1023 : tensor<1x196x512xf32>
    %1025 = stablehlo.add %983, %1024 : tensor<1x196x512xf32>
    %1026 = stablehlo.reduce(%1025 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1027 = stablehlo.divide %1026, %cst_283 : tensor<1x196xf32>
    %1028 = stablehlo.broadcast_in_dim %1027, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1029 = stablehlo.reshape %1028 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1030 = stablehlo.broadcast_in_dim %1029, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1031 = stablehlo.subtract %1025, %1030 : tensor<1x196x512xf32>
    %1032 = stablehlo.multiply %1025, %1025 : tensor<1x196x512xf32>
    %1033 = stablehlo.reduce(%1032 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1034 = stablehlo.divide %1033, %cst_283 : tensor<1x196xf32>
    %1035 = stablehlo.multiply %1027, %1027 : tensor<1x196xf32>
    %1036 = stablehlo.subtract %1034, %1035 : tensor<1x196xf32>
    %1037 = stablehlo.maximum %cst_288, %1036 : tensor<1x196xf32>
    %1038 = stablehlo.broadcast_in_dim %1037, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1039 = stablehlo.add %1038, %cst_287 : tensor<1x196x1xf32>
    %1040 = stablehlo.rsqrt %1039 : tensor<1x196x1xf32>
    %1041 = stablehlo.reshape %1040 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1042 = stablehlo.broadcast_in_dim %1041, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1043 = stablehlo.reshape %cst_238 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1044 = stablehlo.broadcast_in_dim %1043, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1045 = stablehlo.multiply %1042, %1044 : tensor<1x196x512xf32>
    %1046 = stablehlo.multiply %1031, %1045 : tensor<1x196x512xf32>
    %1047 = stablehlo.reshape %cst_237 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1048 = stablehlo.broadcast_in_dim %1047, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1049 = stablehlo.add %1046, %1048 : tensor<1x196x512xf32>
    %1050 = stablehlo.reshape %1049 : (tensor<1x196x512xf32>) -> tensor<1x14x14x512xf32>
    %1051 = stablehlo.slice %1050 [0:1, 3:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %1052 = stablehlo.slice %1050 [0:1, 0:3, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %1053 = stablehlo.concatenate %1051, %1052, dim = 1 : (tensor<1x11x14x512xf32>, tensor<1x3x14x512xf32>) -> tensor<1x14x14x512xf32>
    %1054 = stablehlo.slice %1053 [0:1, 0:14, 3:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %1055 = stablehlo.slice %1053 [0:1, 0:14, 0:3, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %1056 = stablehlo.concatenate %1054, %1055, dim = 2 : (tensor<1x14x11x512xf32>, tensor<1x14x3x512xf32>) -> tensor<1x14x14x512xf32>
    %1057 = stablehlo.reshape %1056 : (tensor<1x14x14x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %1058 = stablehlo.transpose %1057, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %1059 = stablehlo.reshape %1058 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %1060 = stablehlo.dot_general %1059, %cst_236, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %1061 = stablehlo.reshape %cst_235 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %1062 = stablehlo.broadcast_in_dim %1061, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %1063 = stablehlo.add %1060, %1062 : tensor<4x49x1536xf32>
    %1064 = stablehlo.reshape %1063 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %1065 = stablehlo.transpose %1064, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %1066 = stablehlo.slice %1065 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1067 = stablehlo.reshape %1066 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1068 = stablehlo.multiply %1067, %cst_278 : tensor<4x16x49x32xf32>
    %1069 = stablehlo.slice %1065 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1070 = stablehlo.reshape %1069 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1071 = stablehlo.transpose %1070, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %1072 = stablehlo.dot_general %1068, %1071, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %1073 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %1074 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %1075 = stablehlo.select %1073, %1074, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %1076 = stablehlo.broadcast_in_dim %1075, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %1077 = "stablehlo.gather"(%cst_234, %1076) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %1078 = stablehlo.reshape %1077 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %1079 = stablehlo.transpose %1078, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %1080 = stablehlo.broadcast_in_dim %1079, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %1081 = stablehlo.reshape %1080 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %1082 = stablehlo.broadcast_in_dim %1081, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %1083 = stablehlo.add %1072, %1082 : tensor<4x16x49x49xf32>
    %1084 = stablehlo.reduce(%1083 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %1085 = stablehlo.broadcast_in_dim %1084, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %1086 = stablehlo.reshape %1085 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %1087 = stablehlo.broadcast_in_dim %1086, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %1088 = stablehlo.subtract %1083, %1087 : tensor<4x16x49x49xf32>
    %1089 = stablehlo.exponential %1088 : tensor<4x16x49x49xf32>
    %1090 = stablehlo.reduce(%1089 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %1091 = stablehlo.broadcast_in_dim %1090, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %1092 = stablehlo.reshape %1091 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %1093 = stablehlo.broadcast_in_dim %1092, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %1094 = stablehlo.divide %1089, %1093 : tensor<4x16x49x49xf32>
    %1095 = stablehlo.slice %1065 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1096 = stablehlo.reshape %1095 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1097 = stablehlo.dot_general %1094, %1096, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1098 = stablehlo.transpose %1097, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %1099 = stablehlo.reshape %1098 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %1100 = stablehlo.dot_general %1099, %cst_233, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %1101 = stablehlo.reshape %cst_232 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %1102 = stablehlo.broadcast_in_dim %1101, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %1103 = stablehlo.add %1100, %1102 : tensor<4x49x512xf32>
    %1104 = stablehlo.reshape %1103 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %1105 = stablehlo.transpose %1104, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %1106 = stablehlo.reshape %1105 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x14x14x512xf32>
    %1107 = stablehlo.slice %1106 [0:1, 11:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %1108 = stablehlo.slice %1106 [0:1, 0:11, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %1109 = stablehlo.concatenate %1107, %1108, dim = 1 : (tensor<1x3x14x512xf32>, tensor<1x11x14x512xf32>) -> tensor<1x14x14x512xf32>
    %1110 = stablehlo.slice %1109 [0:1, 0:14, 11:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %1111 = stablehlo.slice %1109 [0:1, 0:14, 0:11, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %1112 = stablehlo.concatenate %1110, %1111, dim = 2 : (tensor<1x14x3x512xf32>, tensor<1x14x11x512xf32>) -> tensor<1x14x14x512xf32>
    %1113 = stablehlo.reshape %1112 : (tensor<1x14x14x512xf32>) -> tensor<1x196x512xf32>
    %1114 = stablehlo.add %1025, %1113 : tensor<1x196x512xf32>
    %1115 = stablehlo.reduce(%1114 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1116 = stablehlo.divide %1115, %cst_283 : tensor<1x196xf32>
    %1117 = stablehlo.broadcast_in_dim %1116, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1118 = stablehlo.reshape %1117 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1119 = stablehlo.broadcast_in_dim %1118, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1120 = stablehlo.subtract %1114, %1119 : tensor<1x196x512xf32>
    %1121 = stablehlo.multiply %1114, %1114 : tensor<1x196x512xf32>
    %1122 = stablehlo.reduce(%1121 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1123 = stablehlo.divide %1122, %cst_283 : tensor<1x196xf32>
    %1124 = stablehlo.multiply %1116, %1116 : tensor<1x196xf32>
    %1125 = stablehlo.subtract %1123, %1124 : tensor<1x196xf32>
    %1126 = stablehlo.maximum %cst_288, %1125 : tensor<1x196xf32>
    %1127 = stablehlo.broadcast_in_dim %1126, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1128 = stablehlo.add %1127, %cst_287 : tensor<1x196x1xf32>
    %1129 = stablehlo.rsqrt %1128 : tensor<1x196x1xf32>
    %1130 = stablehlo.reshape %1129 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1131 = stablehlo.broadcast_in_dim %1130, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1132 = stablehlo.reshape %cst_231 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1133 = stablehlo.broadcast_in_dim %1132, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1134 = stablehlo.multiply %1131, %1133 : tensor<1x196x512xf32>
    %1135 = stablehlo.multiply %1120, %1134 : tensor<1x196x512xf32>
    %1136 = stablehlo.reshape %cst_230 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1137 = stablehlo.broadcast_in_dim %1136, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1138 = stablehlo.add %1135, %1137 : tensor<1x196x512xf32>
    %1139 = stablehlo.dot_general %1138, %cst_229, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %1140 = stablehlo.reshape %cst_228 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %1141 = stablehlo.broadcast_in_dim %1140, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %1142 = stablehlo.add %1139, %1141 : tensor<1x196x2048xf32>
    %1143 = stablehlo.multiply %1142, %1142 : tensor<1x196x2048xf32>
    %1144 = stablehlo.multiply %1143, %1142 : tensor<1x196x2048xf32>
    %1145 = stablehlo.multiply %cst_267, %1144 : tensor<1x196x2048xf32>
    %1146 = stablehlo.add %1142, %1145 : tensor<1x196x2048xf32>
    %1147 = stablehlo.multiply %cst_268, %1146 : tensor<1x196x2048xf32>
    %1148 = stablehlo.tanh %1147 : tensor<1x196x2048xf32>
    %1149 = stablehlo.add %cst_269, %1148 : tensor<1x196x2048xf32>
    %1150 = stablehlo.multiply %cst_270, %1149 : tensor<1x196x2048xf32>
    %1151 = stablehlo.multiply %1142, %1150 : tensor<1x196x2048xf32>
    %1152 = stablehlo.dot_general %1151, %cst_227, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %1153 = stablehlo.reshape %cst_226 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1154 = stablehlo.broadcast_in_dim %1153, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1155 = stablehlo.add %1152, %1154 : tensor<1x196x512xf32>
    %1156 = stablehlo.add %1114, %1155 : tensor<1x196x512xf32>
    %1157 = stablehlo.reduce(%1156 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1158 = stablehlo.divide %1157, %cst_283 : tensor<1x196xf32>
    %1159 = stablehlo.broadcast_in_dim %1158, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1160 = stablehlo.reshape %1159 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1161 = stablehlo.broadcast_in_dim %1160, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1162 = stablehlo.subtract %1156, %1161 : tensor<1x196x512xf32>
    %1163 = stablehlo.multiply %1156, %1156 : tensor<1x196x512xf32>
    %1164 = stablehlo.reduce(%1163 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1165 = stablehlo.divide %1164, %cst_283 : tensor<1x196xf32>
    %1166 = stablehlo.multiply %1158, %1158 : tensor<1x196xf32>
    %1167 = stablehlo.subtract %1165, %1166 : tensor<1x196xf32>
    %1168 = stablehlo.maximum %cst_288, %1167 : tensor<1x196xf32>
    %1169 = stablehlo.broadcast_in_dim %1168, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1170 = stablehlo.add %1169, %cst_287 : tensor<1x196x1xf32>
    %1171 = stablehlo.rsqrt %1170 : tensor<1x196x1xf32>
    %1172 = stablehlo.reshape %1171 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1173 = stablehlo.broadcast_in_dim %1172, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1174 = stablehlo.reshape %cst_225 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1175 = stablehlo.broadcast_in_dim %1174, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1176 = stablehlo.multiply %1173, %1175 : tensor<1x196x512xf32>
    %1177 = stablehlo.multiply %1162, %1176 : tensor<1x196x512xf32>
    %1178 = stablehlo.reshape %cst_224 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1179 = stablehlo.broadcast_in_dim %1178, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1180 = stablehlo.add %1177, %1179 : tensor<1x196x512xf32>
    %1181 = stablehlo.reshape %1180 : (tensor<1x196x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %1182 = stablehlo.transpose %1181, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %1183 = stablehlo.reshape %1182 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %1184 = stablehlo.dot_general %1183, %cst_223, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %1185 = stablehlo.reshape %cst_222 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %1186 = stablehlo.broadcast_in_dim %1185, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %1187 = stablehlo.add %1184, %1186 : tensor<4x49x1536xf32>
    %1188 = stablehlo.reshape %1187 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %1189 = stablehlo.transpose %1188, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %1190 = stablehlo.slice %1189 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1191 = stablehlo.reshape %1190 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1192 = stablehlo.multiply %1191, %cst_278 : tensor<4x16x49x32xf32>
    %1193 = stablehlo.slice %1189 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1194 = stablehlo.reshape %1193 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1195 = stablehlo.transpose %1194, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %1196 = stablehlo.dot_general %1192, %1195, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %1197 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %1198 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %1199 = stablehlo.select %1197, %1198, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %1200 = stablehlo.broadcast_in_dim %1199, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %1201 = "stablehlo.gather"(%cst_221, %1200) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %1202 = stablehlo.reshape %1201 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %1203 = stablehlo.transpose %1202, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %1204 = stablehlo.broadcast_in_dim %1203, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %1205 = stablehlo.reshape %1204 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %1206 = stablehlo.broadcast_in_dim %1205, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %1207 = stablehlo.add %1196, %1206 : tensor<4x16x49x49xf32>
    %1208 = stablehlo.reduce(%1207 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %1209 = stablehlo.broadcast_in_dim %1208, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %1210 = stablehlo.reshape %1209 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %1211 = stablehlo.broadcast_in_dim %1210, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %1212 = stablehlo.subtract %1207, %1211 : tensor<4x16x49x49xf32>
    %1213 = stablehlo.exponential %1212 : tensor<4x16x49x49xf32>
    %1214 = stablehlo.reduce(%1213 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %1215 = stablehlo.broadcast_in_dim %1214, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %1216 = stablehlo.reshape %1215 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %1217 = stablehlo.broadcast_in_dim %1216, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %1218 = stablehlo.divide %1213, %1217 : tensor<4x16x49x49xf32>
    %1219 = stablehlo.slice %1189 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1220 = stablehlo.reshape %1219 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1221 = stablehlo.dot_general %1218, %1220, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1222 = stablehlo.transpose %1221, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %1223 = stablehlo.reshape %1222 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %1224 = stablehlo.dot_general %1223, %cst_220, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %1225 = stablehlo.reshape %cst_219 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %1226 = stablehlo.broadcast_in_dim %1225, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %1227 = stablehlo.add %1224, %1226 : tensor<4x49x512xf32>
    %1228 = stablehlo.reshape %1227 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %1229 = stablehlo.transpose %1228, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %1230 = stablehlo.reshape %1229 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x196x512xf32>
    %1231 = stablehlo.add %1156, %1230 : tensor<1x196x512xf32>
    %1232 = stablehlo.reduce(%1231 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1233 = stablehlo.divide %1232, %cst_283 : tensor<1x196xf32>
    %1234 = stablehlo.broadcast_in_dim %1233, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1235 = stablehlo.reshape %1234 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1236 = stablehlo.broadcast_in_dim %1235, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1237 = stablehlo.subtract %1231, %1236 : tensor<1x196x512xf32>
    %1238 = stablehlo.multiply %1231, %1231 : tensor<1x196x512xf32>
    %1239 = stablehlo.reduce(%1238 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1240 = stablehlo.divide %1239, %cst_283 : tensor<1x196xf32>
    %1241 = stablehlo.multiply %1233, %1233 : tensor<1x196xf32>
    %1242 = stablehlo.subtract %1240, %1241 : tensor<1x196xf32>
    %1243 = stablehlo.maximum %cst_288, %1242 : tensor<1x196xf32>
    %1244 = stablehlo.broadcast_in_dim %1243, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1245 = stablehlo.add %1244, %cst_287 : tensor<1x196x1xf32>
    %1246 = stablehlo.rsqrt %1245 : tensor<1x196x1xf32>
    %1247 = stablehlo.reshape %1246 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1248 = stablehlo.broadcast_in_dim %1247, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1249 = stablehlo.reshape %cst_218 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1250 = stablehlo.broadcast_in_dim %1249, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1251 = stablehlo.multiply %1248, %1250 : tensor<1x196x512xf32>
    %1252 = stablehlo.multiply %1237, %1251 : tensor<1x196x512xf32>
    %1253 = stablehlo.reshape %cst_217 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1254 = stablehlo.broadcast_in_dim %1253, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1255 = stablehlo.add %1252, %1254 : tensor<1x196x512xf32>
    %1256 = stablehlo.dot_general %1255, %cst_216, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %1257 = stablehlo.reshape %cst_215 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %1258 = stablehlo.broadcast_in_dim %1257, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %1259 = stablehlo.add %1256, %1258 : tensor<1x196x2048xf32>
    %1260 = stablehlo.multiply %1259, %1259 : tensor<1x196x2048xf32>
    %1261 = stablehlo.multiply %1260, %1259 : tensor<1x196x2048xf32>
    %1262 = stablehlo.multiply %cst_267, %1261 : tensor<1x196x2048xf32>
    %1263 = stablehlo.add %1259, %1262 : tensor<1x196x2048xf32>
    %1264 = stablehlo.multiply %cst_268, %1263 : tensor<1x196x2048xf32>
    %1265 = stablehlo.tanh %1264 : tensor<1x196x2048xf32>
    %1266 = stablehlo.add %cst_269, %1265 : tensor<1x196x2048xf32>
    %1267 = stablehlo.multiply %cst_270, %1266 : tensor<1x196x2048xf32>
    %1268 = stablehlo.multiply %1259, %1267 : tensor<1x196x2048xf32>
    %1269 = stablehlo.dot_general %1268, %cst_214, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %1270 = stablehlo.reshape %cst_213 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1271 = stablehlo.broadcast_in_dim %1270, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1272 = stablehlo.add %1269, %1271 : tensor<1x196x512xf32>
    %1273 = stablehlo.add %1231, %1272 : tensor<1x196x512xf32>
    %1274 = stablehlo.reduce(%1273 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1275 = stablehlo.divide %1274, %cst_283 : tensor<1x196xf32>
    %1276 = stablehlo.broadcast_in_dim %1275, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1277 = stablehlo.reshape %1276 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1278 = stablehlo.broadcast_in_dim %1277, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1279 = stablehlo.subtract %1273, %1278 : tensor<1x196x512xf32>
    %1280 = stablehlo.multiply %1273, %1273 : tensor<1x196x512xf32>
    %1281 = stablehlo.reduce(%1280 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1282 = stablehlo.divide %1281, %cst_283 : tensor<1x196xf32>
    %1283 = stablehlo.multiply %1275, %1275 : tensor<1x196xf32>
    %1284 = stablehlo.subtract %1282, %1283 : tensor<1x196xf32>
    %1285 = stablehlo.maximum %cst_288, %1284 : tensor<1x196xf32>
    %1286 = stablehlo.broadcast_in_dim %1285, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1287 = stablehlo.add %1286, %cst_287 : tensor<1x196x1xf32>
    %1288 = stablehlo.rsqrt %1287 : tensor<1x196x1xf32>
    %1289 = stablehlo.reshape %1288 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1290 = stablehlo.broadcast_in_dim %1289, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1291 = stablehlo.reshape %cst_212 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1292 = stablehlo.broadcast_in_dim %1291, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1293 = stablehlo.multiply %1290, %1292 : tensor<1x196x512xf32>
    %1294 = stablehlo.multiply %1279, %1293 : tensor<1x196x512xf32>
    %1295 = stablehlo.reshape %cst_211 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1296 = stablehlo.broadcast_in_dim %1295, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1297 = stablehlo.add %1294, %1296 : tensor<1x196x512xf32>
    %1298 = stablehlo.reshape %1297 : (tensor<1x196x512xf32>) -> tensor<1x14x14x512xf32>
    %1299 = stablehlo.slice %1298 [0:1, 3:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %1300 = stablehlo.slice %1298 [0:1, 0:3, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %1301 = stablehlo.concatenate %1299, %1300, dim = 1 : (tensor<1x11x14x512xf32>, tensor<1x3x14x512xf32>) -> tensor<1x14x14x512xf32>
    %1302 = stablehlo.slice %1301 [0:1, 0:14, 3:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %1303 = stablehlo.slice %1301 [0:1, 0:14, 0:3, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %1304 = stablehlo.concatenate %1302, %1303, dim = 2 : (tensor<1x14x11x512xf32>, tensor<1x14x3x512xf32>) -> tensor<1x14x14x512xf32>
    %1305 = stablehlo.reshape %1304 : (tensor<1x14x14x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %1306 = stablehlo.transpose %1305, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %1307 = stablehlo.reshape %1306 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %1308 = stablehlo.dot_general %1307, %cst_210, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %1309 = stablehlo.reshape %cst_209 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %1310 = stablehlo.broadcast_in_dim %1309, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %1311 = stablehlo.add %1308, %1310 : tensor<4x49x1536xf32>
    %1312 = stablehlo.reshape %1311 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %1313 = stablehlo.transpose %1312, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %1314 = stablehlo.slice %1313 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1315 = stablehlo.reshape %1314 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1316 = stablehlo.multiply %1315, %cst_278 : tensor<4x16x49x32xf32>
    %1317 = stablehlo.slice %1313 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1318 = stablehlo.reshape %1317 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1319 = stablehlo.transpose %1318, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %1320 = stablehlo.dot_general %1316, %1319, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %1321 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %1322 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %1323 = stablehlo.select %1321, %1322, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %1324 = stablehlo.broadcast_in_dim %1323, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %1325 = "stablehlo.gather"(%cst_208, %1324) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %1326 = stablehlo.reshape %1325 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %1327 = stablehlo.transpose %1326, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %1328 = stablehlo.broadcast_in_dim %1327, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %1329 = stablehlo.reshape %1328 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %1330 = stablehlo.broadcast_in_dim %1329, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %1331 = stablehlo.add %1320, %1330 : tensor<4x16x49x49xf32>
    %1332 = stablehlo.reduce(%1331 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %1333 = stablehlo.broadcast_in_dim %1332, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %1334 = stablehlo.reshape %1333 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %1335 = stablehlo.broadcast_in_dim %1334, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %1336 = stablehlo.subtract %1331, %1335 : tensor<4x16x49x49xf32>
    %1337 = stablehlo.exponential %1336 : tensor<4x16x49x49xf32>
    %1338 = stablehlo.reduce(%1337 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %1339 = stablehlo.broadcast_in_dim %1338, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %1340 = stablehlo.reshape %1339 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %1341 = stablehlo.broadcast_in_dim %1340, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %1342 = stablehlo.divide %1337, %1341 : tensor<4x16x49x49xf32>
    %1343 = stablehlo.slice %1313 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1344 = stablehlo.reshape %1343 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1345 = stablehlo.dot_general %1342, %1344, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1346 = stablehlo.transpose %1345, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %1347 = stablehlo.reshape %1346 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %1348 = stablehlo.dot_general %1347, %cst_207, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %1349 = stablehlo.reshape %cst_206 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %1350 = stablehlo.broadcast_in_dim %1349, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %1351 = stablehlo.add %1348, %1350 : tensor<4x49x512xf32>
    %1352 = stablehlo.reshape %1351 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %1353 = stablehlo.transpose %1352, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %1354 = stablehlo.reshape %1353 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x14x14x512xf32>
    %1355 = stablehlo.slice %1354 [0:1, 11:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %1356 = stablehlo.slice %1354 [0:1, 0:11, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %1357 = stablehlo.concatenate %1355, %1356, dim = 1 : (tensor<1x3x14x512xf32>, tensor<1x11x14x512xf32>) -> tensor<1x14x14x512xf32>
    %1358 = stablehlo.slice %1357 [0:1, 0:14, 11:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %1359 = stablehlo.slice %1357 [0:1, 0:14, 0:11, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %1360 = stablehlo.concatenate %1358, %1359, dim = 2 : (tensor<1x14x3x512xf32>, tensor<1x14x11x512xf32>) -> tensor<1x14x14x512xf32>
    %1361 = stablehlo.reshape %1360 : (tensor<1x14x14x512xf32>) -> tensor<1x196x512xf32>
    %1362 = stablehlo.add %1273, %1361 : tensor<1x196x512xf32>
    %1363 = stablehlo.reduce(%1362 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1364 = stablehlo.divide %1363, %cst_283 : tensor<1x196xf32>
    %1365 = stablehlo.broadcast_in_dim %1364, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1366 = stablehlo.reshape %1365 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1367 = stablehlo.broadcast_in_dim %1366, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1368 = stablehlo.subtract %1362, %1367 : tensor<1x196x512xf32>
    %1369 = stablehlo.multiply %1362, %1362 : tensor<1x196x512xf32>
    %1370 = stablehlo.reduce(%1369 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1371 = stablehlo.divide %1370, %cst_283 : tensor<1x196xf32>
    %1372 = stablehlo.multiply %1364, %1364 : tensor<1x196xf32>
    %1373 = stablehlo.subtract %1371, %1372 : tensor<1x196xf32>
    %1374 = stablehlo.maximum %cst_288, %1373 : tensor<1x196xf32>
    %1375 = stablehlo.broadcast_in_dim %1374, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1376 = stablehlo.add %1375, %cst_287 : tensor<1x196x1xf32>
    %1377 = stablehlo.rsqrt %1376 : tensor<1x196x1xf32>
    %1378 = stablehlo.reshape %1377 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1379 = stablehlo.broadcast_in_dim %1378, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1380 = stablehlo.reshape %cst_205 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1381 = stablehlo.broadcast_in_dim %1380, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1382 = stablehlo.multiply %1379, %1381 : tensor<1x196x512xf32>
    %1383 = stablehlo.multiply %1368, %1382 : tensor<1x196x512xf32>
    %1384 = stablehlo.reshape %cst_204 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1385 = stablehlo.broadcast_in_dim %1384, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1386 = stablehlo.add %1383, %1385 : tensor<1x196x512xf32>
    %1387 = stablehlo.dot_general %1386, %cst_203, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %1388 = stablehlo.reshape %cst_202 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %1389 = stablehlo.broadcast_in_dim %1388, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %1390 = stablehlo.add %1387, %1389 : tensor<1x196x2048xf32>
    %1391 = stablehlo.multiply %1390, %1390 : tensor<1x196x2048xf32>
    %1392 = stablehlo.multiply %1391, %1390 : tensor<1x196x2048xf32>
    %1393 = stablehlo.multiply %cst_267, %1392 : tensor<1x196x2048xf32>
    %1394 = stablehlo.add %1390, %1393 : tensor<1x196x2048xf32>
    %1395 = stablehlo.multiply %cst_268, %1394 : tensor<1x196x2048xf32>
    %1396 = stablehlo.tanh %1395 : tensor<1x196x2048xf32>
    %1397 = stablehlo.add %cst_269, %1396 : tensor<1x196x2048xf32>
    %1398 = stablehlo.multiply %cst_270, %1397 : tensor<1x196x2048xf32>
    %1399 = stablehlo.multiply %1390, %1398 : tensor<1x196x2048xf32>
    %1400 = stablehlo.dot_general %1399, %cst_201, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %1401 = stablehlo.reshape %cst_200 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1402 = stablehlo.broadcast_in_dim %1401, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1403 = stablehlo.add %1400, %1402 : tensor<1x196x512xf32>
    %1404 = stablehlo.add %1362, %1403 : tensor<1x196x512xf32>
    %1405 = stablehlo.reduce(%1404 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1406 = stablehlo.divide %1405, %cst_283 : tensor<1x196xf32>
    %1407 = stablehlo.broadcast_in_dim %1406, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1408 = stablehlo.reshape %1407 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1409 = stablehlo.broadcast_in_dim %1408, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1410 = stablehlo.subtract %1404, %1409 : tensor<1x196x512xf32>
    %1411 = stablehlo.multiply %1404, %1404 : tensor<1x196x512xf32>
    %1412 = stablehlo.reduce(%1411 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1413 = stablehlo.divide %1412, %cst_283 : tensor<1x196xf32>
    %1414 = stablehlo.multiply %1406, %1406 : tensor<1x196xf32>
    %1415 = stablehlo.subtract %1413, %1414 : tensor<1x196xf32>
    %1416 = stablehlo.maximum %cst_288, %1415 : tensor<1x196xf32>
    %1417 = stablehlo.broadcast_in_dim %1416, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1418 = stablehlo.add %1417, %cst_287 : tensor<1x196x1xf32>
    %1419 = stablehlo.rsqrt %1418 : tensor<1x196x1xf32>
    %1420 = stablehlo.reshape %1419 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1421 = stablehlo.broadcast_in_dim %1420, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1422 = stablehlo.reshape %cst_199 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1423 = stablehlo.broadcast_in_dim %1422, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1424 = stablehlo.multiply %1421, %1423 : tensor<1x196x512xf32>
    %1425 = stablehlo.multiply %1410, %1424 : tensor<1x196x512xf32>
    %1426 = stablehlo.reshape %cst_198 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1427 = stablehlo.broadcast_in_dim %1426, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1428 = stablehlo.add %1425, %1427 : tensor<1x196x512xf32>
    %1429 = stablehlo.reshape %1428 : (tensor<1x196x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %1430 = stablehlo.transpose %1429, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %1431 = stablehlo.reshape %1430 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %1432 = stablehlo.dot_general %1431, %cst_197, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %1433 = stablehlo.reshape %cst_196 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %1434 = stablehlo.broadcast_in_dim %1433, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %1435 = stablehlo.add %1432, %1434 : tensor<4x49x1536xf32>
    %1436 = stablehlo.reshape %1435 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %1437 = stablehlo.transpose %1436, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %1438 = stablehlo.slice %1437 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1439 = stablehlo.reshape %1438 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1440 = stablehlo.multiply %1439, %cst_278 : tensor<4x16x49x32xf32>
    %1441 = stablehlo.slice %1437 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1442 = stablehlo.reshape %1441 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1443 = stablehlo.transpose %1442, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %1444 = stablehlo.dot_general %1440, %1443, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %1445 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %1446 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %1447 = stablehlo.select %1445, %1446, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %1448 = stablehlo.broadcast_in_dim %1447, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %1449 = "stablehlo.gather"(%cst_195, %1448) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %1450 = stablehlo.reshape %1449 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %1451 = stablehlo.transpose %1450, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %1452 = stablehlo.broadcast_in_dim %1451, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %1453 = stablehlo.reshape %1452 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %1454 = stablehlo.broadcast_in_dim %1453, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %1455 = stablehlo.add %1444, %1454 : tensor<4x16x49x49xf32>
    %1456 = stablehlo.reduce(%1455 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %1457 = stablehlo.broadcast_in_dim %1456, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %1458 = stablehlo.reshape %1457 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %1459 = stablehlo.broadcast_in_dim %1458, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %1460 = stablehlo.subtract %1455, %1459 : tensor<4x16x49x49xf32>
    %1461 = stablehlo.exponential %1460 : tensor<4x16x49x49xf32>
    %1462 = stablehlo.reduce(%1461 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %1463 = stablehlo.broadcast_in_dim %1462, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %1464 = stablehlo.reshape %1463 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %1465 = stablehlo.broadcast_in_dim %1464, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %1466 = stablehlo.divide %1461, %1465 : tensor<4x16x49x49xf32>
    %1467 = stablehlo.slice %1437 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1468 = stablehlo.reshape %1467 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1469 = stablehlo.dot_general %1466, %1468, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1470 = stablehlo.transpose %1469, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %1471 = stablehlo.reshape %1470 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %1472 = stablehlo.dot_general %1471, %cst_194, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %1473 = stablehlo.reshape %cst_193 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %1474 = stablehlo.broadcast_in_dim %1473, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %1475 = stablehlo.add %1472, %1474 : tensor<4x49x512xf32>
    %1476 = stablehlo.reshape %1475 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %1477 = stablehlo.transpose %1476, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %1478 = stablehlo.reshape %1477 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x196x512xf32>
    %1479 = stablehlo.add %1404, %1478 : tensor<1x196x512xf32>
    %1480 = stablehlo.reduce(%1479 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1481 = stablehlo.divide %1480, %cst_283 : tensor<1x196xf32>
    %1482 = stablehlo.broadcast_in_dim %1481, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1483 = stablehlo.reshape %1482 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1484 = stablehlo.broadcast_in_dim %1483, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1485 = stablehlo.subtract %1479, %1484 : tensor<1x196x512xf32>
    %1486 = stablehlo.multiply %1479, %1479 : tensor<1x196x512xf32>
    %1487 = stablehlo.reduce(%1486 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1488 = stablehlo.divide %1487, %cst_283 : tensor<1x196xf32>
    %1489 = stablehlo.multiply %1481, %1481 : tensor<1x196xf32>
    %1490 = stablehlo.subtract %1488, %1489 : tensor<1x196xf32>
    %1491 = stablehlo.maximum %cst_288, %1490 : tensor<1x196xf32>
    %1492 = stablehlo.broadcast_in_dim %1491, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1493 = stablehlo.add %1492, %cst_287 : tensor<1x196x1xf32>
    %1494 = stablehlo.rsqrt %1493 : tensor<1x196x1xf32>
    %1495 = stablehlo.reshape %1494 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1496 = stablehlo.broadcast_in_dim %1495, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1497 = stablehlo.reshape %cst_192 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1498 = stablehlo.broadcast_in_dim %1497, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1499 = stablehlo.multiply %1496, %1498 : tensor<1x196x512xf32>
    %1500 = stablehlo.multiply %1485, %1499 : tensor<1x196x512xf32>
    %1501 = stablehlo.reshape %cst_191 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1502 = stablehlo.broadcast_in_dim %1501, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1503 = stablehlo.add %1500, %1502 : tensor<1x196x512xf32>
    %1504 = stablehlo.dot_general %1503, %cst_190, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %1505 = stablehlo.reshape %cst_189 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %1506 = stablehlo.broadcast_in_dim %1505, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %1507 = stablehlo.add %1504, %1506 : tensor<1x196x2048xf32>
    %1508 = stablehlo.multiply %1507, %1507 : tensor<1x196x2048xf32>
    %1509 = stablehlo.multiply %1508, %1507 : tensor<1x196x2048xf32>
    %1510 = stablehlo.multiply %cst_267, %1509 : tensor<1x196x2048xf32>
    %1511 = stablehlo.add %1507, %1510 : tensor<1x196x2048xf32>
    %1512 = stablehlo.multiply %cst_268, %1511 : tensor<1x196x2048xf32>
    %1513 = stablehlo.tanh %1512 : tensor<1x196x2048xf32>
    %1514 = stablehlo.add %cst_269, %1513 : tensor<1x196x2048xf32>
    %1515 = stablehlo.multiply %cst_270, %1514 : tensor<1x196x2048xf32>
    %1516 = stablehlo.multiply %1507, %1515 : tensor<1x196x2048xf32>
    %1517 = stablehlo.dot_general %1516, %cst_188, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %1518 = stablehlo.reshape %cst_187 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1519 = stablehlo.broadcast_in_dim %1518, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1520 = stablehlo.add %1517, %1519 : tensor<1x196x512xf32>
    %1521 = stablehlo.add %1479, %1520 : tensor<1x196x512xf32>
    %1522 = stablehlo.reduce(%1521 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1523 = stablehlo.divide %1522, %cst_283 : tensor<1x196xf32>
    %1524 = stablehlo.broadcast_in_dim %1523, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1525 = stablehlo.reshape %1524 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1526 = stablehlo.broadcast_in_dim %1525, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1527 = stablehlo.subtract %1521, %1526 : tensor<1x196x512xf32>
    %1528 = stablehlo.multiply %1521, %1521 : tensor<1x196x512xf32>
    %1529 = stablehlo.reduce(%1528 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1530 = stablehlo.divide %1529, %cst_283 : tensor<1x196xf32>
    %1531 = stablehlo.multiply %1523, %1523 : tensor<1x196xf32>
    %1532 = stablehlo.subtract %1530, %1531 : tensor<1x196xf32>
    %1533 = stablehlo.maximum %cst_288, %1532 : tensor<1x196xf32>
    %1534 = stablehlo.broadcast_in_dim %1533, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1535 = stablehlo.add %1534, %cst_287 : tensor<1x196x1xf32>
    %1536 = stablehlo.rsqrt %1535 : tensor<1x196x1xf32>
    %1537 = stablehlo.reshape %1536 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1538 = stablehlo.broadcast_in_dim %1537, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1539 = stablehlo.reshape %cst_186 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1540 = stablehlo.broadcast_in_dim %1539, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1541 = stablehlo.multiply %1538, %1540 : tensor<1x196x512xf32>
    %1542 = stablehlo.multiply %1527, %1541 : tensor<1x196x512xf32>
    %1543 = stablehlo.reshape %cst_185 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1544 = stablehlo.broadcast_in_dim %1543, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1545 = stablehlo.add %1542, %1544 : tensor<1x196x512xf32>
    %1546 = stablehlo.reshape %1545 : (tensor<1x196x512xf32>) -> tensor<1x14x14x512xf32>
    %1547 = stablehlo.slice %1546 [0:1, 3:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %1548 = stablehlo.slice %1546 [0:1, 0:3, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %1549 = stablehlo.concatenate %1547, %1548, dim = 1 : (tensor<1x11x14x512xf32>, tensor<1x3x14x512xf32>) -> tensor<1x14x14x512xf32>
    %1550 = stablehlo.slice %1549 [0:1, 0:14, 3:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %1551 = stablehlo.slice %1549 [0:1, 0:14, 0:3, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %1552 = stablehlo.concatenate %1550, %1551, dim = 2 : (tensor<1x14x11x512xf32>, tensor<1x14x3x512xf32>) -> tensor<1x14x14x512xf32>
    %1553 = stablehlo.reshape %1552 : (tensor<1x14x14x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %1554 = stablehlo.transpose %1553, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %1555 = stablehlo.reshape %1554 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %1556 = stablehlo.dot_general %1555, %cst_184, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %1557 = stablehlo.reshape %cst_183 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %1558 = stablehlo.broadcast_in_dim %1557, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %1559 = stablehlo.add %1556, %1558 : tensor<4x49x1536xf32>
    %1560 = stablehlo.reshape %1559 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %1561 = stablehlo.transpose %1560, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %1562 = stablehlo.slice %1561 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1563 = stablehlo.reshape %1562 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1564 = stablehlo.multiply %1563, %cst_278 : tensor<4x16x49x32xf32>
    %1565 = stablehlo.slice %1561 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1566 = stablehlo.reshape %1565 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1567 = stablehlo.transpose %1566, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %1568 = stablehlo.dot_general %1564, %1567, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %1569 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %1570 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %1571 = stablehlo.select %1569, %1570, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %1572 = stablehlo.broadcast_in_dim %1571, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %1573 = "stablehlo.gather"(%cst_182, %1572) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %1574 = stablehlo.reshape %1573 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %1575 = stablehlo.transpose %1574, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %1576 = stablehlo.broadcast_in_dim %1575, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %1577 = stablehlo.reshape %1576 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %1578 = stablehlo.broadcast_in_dim %1577, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %1579 = stablehlo.add %1568, %1578 : tensor<4x16x49x49xf32>
    %1580 = stablehlo.reduce(%1579 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %1581 = stablehlo.broadcast_in_dim %1580, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %1582 = stablehlo.reshape %1581 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %1583 = stablehlo.broadcast_in_dim %1582, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %1584 = stablehlo.subtract %1579, %1583 : tensor<4x16x49x49xf32>
    %1585 = stablehlo.exponential %1584 : tensor<4x16x49x49xf32>
    %1586 = stablehlo.reduce(%1585 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %1587 = stablehlo.broadcast_in_dim %1586, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %1588 = stablehlo.reshape %1587 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %1589 = stablehlo.broadcast_in_dim %1588, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %1590 = stablehlo.divide %1585, %1589 : tensor<4x16x49x49xf32>
    %1591 = stablehlo.slice %1561 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1592 = stablehlo.reshape %1591 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1593 = stablehlo.dot_general %1590, %1592, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1594 = stablehlo.transpose %1593, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %1595 = stablehlo.reshape %1594 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %1596 = stablehlo.dot_general %1595, %cst_181, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %1597 = stablehlo.reshape %cst_180 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %1598 = stablehlo.broadcast_in_dim %1597, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %1599 = stablehlo.add %1596, %1598 : tensor<4x49x512xf32>
    %1600 = stablehlo.reshape %1599 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %1601 = stablehlo.transpose %1600, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %1602 = stablehlo.reshape %1601 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x14x14x512xf32>
    %1603 = stablehlo.slice %1602 [0:1, 11:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %1604 = stablehlo.slice %1602 [0:1, 0:11, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %1605 = stablehlo.concatenate %1603, %1604, dim = 1 : (tensor<1x3x14x512xf32>, tensor<1x11x14x512xf32>) -> tensor<1x14x14x512xf32>
    %1606 = stablehlo.slice %1605 [0:1, 0:14, 11:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %1607 = stablehlo.slice %1605 [0:1, 0:14, 0:11, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %1608 = stablehlo.concatenate %1606, %1607, dim = 2 : (tensor<1x14x3x512xf32>, tensor<1x14x11x512xf32>) -> tensor<1x14x14x512xf32>
    %1609 = stablehlo.reshape %1608 : (tensor<1x14x14x512xf32>) -> tensor<1x196x512xf32>
    %1610 = stablehlo.add %1521, %1609 : tensor<1x196x512xf32>
    %1611 = stablehlo.reduce(%1610 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1612 = stablehlo.divide %1611, %cst_283 : tensor<1x196xf32>
    %1613 = stablehlo.broadcast_in_dim %1612, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1614 = stablehlo.reshape %1613 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1615 = stablehlo.broadcast_in_dim %1614, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1616 = stablehlo.subtract %1610, %1615 : tensor<1x196x512xf32>
    %1617 = stablehlo.multiply %1610, %1610 : tensor<1x196x512xf32>
    %1618 = stablehlo.reduce(%1617 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1619 = stablehlo.divide %1618, %cst_283 : tensor<1x196xf32>
    %1620 = stablehlo.multiply %1612, %1612 : tensor<1x196xf32>
    %1621 = stablehlo.subtract %1619, %1620 : tensor<1x196xf32>
    %1622 = stablehlo.maximum %cst_288, %1621 : tensor<1x196xf32>
    %1623 = stablehlo.broadcast_in_dim %1622, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1624 = stablehlo.add %1623, %cst_287 : tensor<1x196x1xf32>
    %1625 = stablehlo.rsqrt %1624 : tensor<1x196x1xf32>
    %1626 = stablehlo.reshape %1625 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1627 = stablehlo.broadcast_in_dim %1626, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1628 = stablehlo.reshape %cst_179 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1629 = stablehlo.broadcast_in_dim %1628, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1630 = stablehlo.multiply %1627, %1629 : tensor<1x196x512xf32>
    %1631 = stablehlo.multiply %1616, %1630 : tensor<1x196x512xf32>
    %1632 = stablehlo.reshape %cst_178 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1633 = stablehlo.broadcast_in_dim %1632, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1634 = stablehlo.add %1631, %1633 : tensor<1x196x512xf32>
    %1635 = stablehlo.dot_general %1634, %cst_177, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %1636 = stablehlo.reshape %cst_176 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %1637 = stablehlo.broadcast_in_dim %1636, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %1638 = stablehlo.add %1635, %1637 : tensor<1x196x2048xf32>
    %1639 = stablehlo.multiply %1638, %1638 : tensor<1x196x2048xf32>
    %1640 = stablehlo.multiply %1639, %1638 : tensor<1x196x2048xf32>
    %1641 = stablehlo.multiply %cst_267, %1640 : tensor<1x196x2048xf32>
    %1642 = stablehlo.add %1638, %1641 : tensor<1x196x2048xf32>
    %1643 = stablehlo.multiply %cst_268, %1642 : tensor<1x196x2048xf32>
    %1644 = stablehlo.tanh %1643 : tensor<1x196x2048xf32>
    %1645 = stablehlo.add %cst_269, %1644 : tensor<1x196x2048xf32>
    %1646 = stablehlo.multiply %cst_270, %1645 : tensor<1x196x2048xf32>
    %1647 = stablehlo.multiply %1638, %1646 : tensor<1x196x2048xf32>
    %1648 = stablehlo.dot_general %1647, %cst_175, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %1649 = stablehlo.reshape %cst_174 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1650 = stablehlo.broadcast_in_dim %1649, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1651 = stablehlo.add %1648, %1650 : tensor<1x196x512xf32>
    %1652 = stablehlo.add %1610, %1651 : tensor<1x196x512xf32>
    %1653 = stablehlo.reduce(%1652 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1654 = stablehlo.divide %1653, %cst_283 : tensor<1x196xf32>
    %1655 = stablehlo.broadcast_in_dim %1654, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1656 = stablehlo.reshape %1655 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1657 = stablehlo.broadcast_in_dim %1656, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1658 = stablehlo.subtract %1652, %1657 : tensor<1x196x512xf32>
    %1659 = stablehlo.multiply %1652, %1652 : tensor<1x196x512xf32>
    %1660 = stablehlo.reduce(%1659 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1661 = stablehlo.divide %1660, %cst_283 : tensor<1x196xf32>
    %1662 = stablehlo.multiply %1654, %1654 : tensor<1x196xf32>
    %1663 = stablehlo.subtract %1661, %1662 : tensor<1x196xf32>
    %1664 = stablehlo.maximum %cst_288, %1663 : tensor<1x196xf32>
    %1665 = stablehlo.broadcast_in_dim %1664, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1666 = stablehlo.add %1665, %cst_287 : tensor<1x196x1xf32>
    %1667 = stablehlo.rsqrt %1666 : tensor<1x196x1xf32>
    %1668 = stablehlo.reshape %1667 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1669 = stablehlo.broadcast_in_dim %1668, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1670 = stablehlo.reshape %cst_173 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1671 = stablehlo.broadcast_in_dim %1670, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1672 = stablehlo.multiply %1669, %1671 : tensor<1x196x512xf32>
    %1673 = stablehlo.multiply %1658, %1672 : tensor<1x196x512xf32>
    %1674 = stablehlo.reshape %cst_172 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1675 = stablehlo.broadcast_in_dim %1674, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1676 = stablehlo.add %1673, %1675 : tensor<1x196x512xf32>
    %1677 = stablehlo.reshape %1676 : (tensor<1x196x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %1678 = stablehlo.transpose %1677, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %1679 = stablehlo.reshape %1678 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %1680 = stablehlo.dot_general %1679, %cst_171, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %1681 = stablehlo.reshape %cst_170 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %1682 = stablehlo.broadcast_in_dim %1681, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %1683 = stablehlo.add %1680, %1682 : tensor<4x49x1536xf32>
    %1684 = stablehlo.reshape %1683 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %1685 = stablehlo.transpose %1684, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %1686 = stablehlo.slice %1685 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1687 = stablehlo.reshape %1686 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1688 = stablehlo.multiply %1687, %cst_278 : tensor<4x16x49x32xf32>
    %1689 = stablehlo.slice %1685 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1690 = stablehlo.reshape %1689 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1691 = stablehlo.transpose %1690, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %1692 = stablehlo.dot_general %1688, %1691, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %1693 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %1694 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %1695 = stablehlo.select %1693, %1694, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %1696 = stablehlo.broadcast_in_dim %1695, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %1697 = "stablehlo.gather"(%cst_169, %1696) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %1698 = stablehlo.reshape %1697 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %1699 = stablehlo.transpose %1698, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %1700 = stablehlo.broadcast_in_dim %1699, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %1701 = stablehlo.reshape %1700 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %1702 = stablehlo.broadcast_in_dim %1701, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %1703 = stablehlo.add %1692, %1702 : tensor<4x16x49x49xf32>
    %1704 = stablehlo.reduce(%1703 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %1705 = stablehlo.broadcast_in_dim %1704, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %1706 = stablehlo.reshape %1705 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %1707 = stablehlo.broadcast_in_dim %1706, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %1708 = stablehlo.subtract %1703, %1707 : tensor<4x16x49x49xf32>
    %1709 = stablehlo.exponential %1708 : tensor<4x16x49x49xf32>
    %1710 = stablehlo.reduce(%1709 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %1711 = stablehlo.broadcast_in_dim %1710, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %1712 = stablehlo.reshape %1711 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %1713 = stablehlo.broadcast_in_dim %1712, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %1714 = stablehlo.divide %1709, %1713 : tensor<4x16x49x49xf32>
    %1715 = stablehlo.slice %1685 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1716 = stablehlo.reshape %1715 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1717 = stablehlo.dot_general %1714, %1716, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1718 = stablehlo.transpose %1717, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %1719 = stablehlo.reshape %1718 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %1720 = stablehlo.dot_general %1719, %cst_168, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %1721 = stablehlo.reshape %cst_167 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %1722 = stablehlo.broadcast_in_dim %1721, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %1723 = stablehlo.add %1720, %1722 : tensor<4x49x512xf32>
    %1724 = stablehlo.reshape %1723 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %1725 = stablehlo.transpose %1724, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %1726 = stablehlo.reshape %1725 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x196x512xf32>
    %1727 = stablehlo.add %1652, %1726 : tensor<1x196x512xf32>
    %1728 = stablehlo.reduce(%1727 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1729 = stablehlo.divide %1728, %cst_283 : tensor<1x196xf32>
    %1730 = stablehlo.broadcast_in_dim %1729, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1731 = stablehlo.reshape %1730 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1732 = stablehlo.broadcast_in_dim %1731, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1733 = stablehlo.subtract %1727, %1732 : tensor<1x196x512xf32>
    %1734 = stablehlo.multiply %1727, %1727 : tensor<1x196x512xf32>
    %1735 = stablehlo.reduce(%1734 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1736 = stablehlo.divide %1735, %cst_283 : tensor<1x196xf32>
    %1737 = stablehlo.multiply %1729, %1729 : tensor<1x196xf32>
    %1738 = stablehlo.subtract %1736, %1737 : tensor<1x196xf32>
    %1739 = stablehlo.maximum %cst_288, %1738 : tensor<1x196xf32>
    %1740 = stablehlo.broadcast_in_dim %1739, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1741 = stablehlo.add %1740, %cst_287 : tensor<1x196x1xf32>
    %1742 = stablehlo.rsqrt %1741 : tensor<1x196x1xf32>
    %1743 = stablehlo.reshape %1742 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1744 = stablehlo.broadcast_in_dim %1743, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1745 = stablehlo.reshape %cst_166 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1746 = stablehlo.broadcast_in_dim %1745, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1747 = stablehlo.multiply %1744, %1746 : tensor<1x196x512xf32>
    %1748 = stablehlo.multiply %1733, %1747 : tensor<1x196x512xf32>
    %1749 = stablehlo.reshape %cst_165 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1750 = stablehlo.broadcast_in_dim %1749, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1751 = stablehlo.add %1748, %1750 : tensor<1x196x512xf32>
    %1752 = stablehlo.dot_general %1751, %cst_164, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %1753 = stablehlo.reshape %cst_163 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %1754 = stablehlo.broadcast_in_dim %1753, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %1755 = stablehlo.add %1752, %1754 : tensor<1x196x2048xf32>
    %1756 = stablehlo.multiply %1755, %1755 : tensor<1x196x2048xf32>
    %1757 = stablehlo.multiply %1756, %1755 : tensor<1x196x2048xf32>
    %1758 = stablehlo.multiply %cst_267, %1757 : tensor<1x196x2048xf32>
    %1759 = stablehlo.add %1755, %1758 : tensor<1x196x2048xf32>
    %1760 = stablehlo.multiply %cst_268, %1759 : tensor<1x196x2048xf32>
    %1761 = stablehlo.tanh %1760 : tensor<1x196x2048xf32>
    %1762 = stablehlo.add %cst_269, %1761 : tensor<1x196x2048xf32>
    %1763 = stablehlo.multiply %cst_270, %1762 : tensor<1x196x2048xf32>
    %1764 = stablehlo.multiply %1755, %1763 : tensor<1x196x2048xf32>
    %1765 = stablehlo.dot_general %1764, %cst_162, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %1766 = stablehlo.reshape %cst_161 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1767 = stablehlo.broadcast_in_dim %1766, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1768 = stablehlo.add %1765, %1767 : tensor<1x196x512xf32>
    %1769 = stablehlo.add %1727, %1768 : tensor<1x196x512xf32>
    %1770 = stablehlo.reduce(%1769 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1771 = stablehlo.divide %1770, %cst_283 : tensor<1x196xf32>
    %1772 = stablehlo.broadcast_in_dim %1771, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1773 = stablehlo.reshape %1772 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1774 = stablehlo.broadcast_in_dim %1773, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1775 = stablehlo.subtract %1769, %1774 : tensor<1x196x512xf32>
    %1776 = stablehlo.multiply %1769, %1769 : tensor<1x196x512xf32>
    %1777 = stablehlo.reduce(%1776 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1778 = stablehlo.divide %1777, %cst_283 : tensor<1x196xf32>
    %1779 = stablehlo.multiply %1771, %1771 : tensor<1x196xf32>
    %1780 = stablehlo.subtract %1778, %1779 : tensor<1x196xf32>
    %1781 = stablehlo.maximum %cst_288, %1780 : tensor<1x196xf32>
    %1782 = stablehlo.broadcast_in_dim %1781, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1783 = stablehlo.add %1782, %cst_287 : tensor<1x196x1xf32>
    %1784 = stablehlo.rsqrt %1783 : tensor<1x196x1xf32>
    %1785 = stablehlo.reshape %1784 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1786 = stablehlo.broadcast_in_dim %1785, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1787 = stablehlo.reshape %cst_160 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1788 = stablehlo.broadcast_in_dim %1787, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1789 = stablehlo.multiply %1786, %1788 : tensor<1x196x512xf32>
    %1790 = stablehlo.multiply %1775, %1789 : tensor<1x196x512xf32>
    %1791 = stablehlo.reshape %cst_159 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1792 = stablehlo.broadcast_in_dim %1791, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1793 = stablehlo.add %1790, %1792 : tensor<1x196x512xf32>
    %1794 = stablehlo.reshape %1793 : (tensor<1x196x512xf32>) -> tensor<1x14x14x512xf32>
    %1795 = stablehlo.slice %1794 [0:1, 3:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %1796 = stablehlo.slice %1794 [0:1, 0:3, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %1797 = stablehlo.concatenate %1795, %1796, dim = 1 : (tensor<1x11x14x512xf32>, tensor<1x3x14x512xf32>) -> tensor<1x14x14x512xf32>
    %1798 = stablehlo.slice %1797 [0:1, 0:14, 3:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %1799 = stablehlo.slice %1797 [0:1, 0:14, 0:3, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %1800 = stablehlo.concatenate %1798, %1799, dim = 2 : (tensor<1x14x11x512xf32>, tensor<1x14x3x512xf32>) -> tensor<1x14x14x512xf32>
    %1801 = stablehlo.reshape %1800 : (tensor<1x14x14x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %1802 = stablehlo.transpose %1801, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %1803 = stablehlo.reshape %1802 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %1804 = stablehlo.dot_general %1803, %cst_158, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %1805 = stablehlo.reshape %cst_157 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %1806 = stablehlo.broadcast_in_dim %1805, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %1807 = stablehlo.add %1804, %1806 : tensor<4x49x1536xf32>
    %1808 = stablehlo.reshape %1807 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %1809 = stablehlo.transpose %1808, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %1810 = stablehlo.slice %1809 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1811 = stablehlo.reshape %1810 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1812 = stablehlo.multiply %1811, %cst_278 : tensor<4x16x49x32xf32>
    %1813 = stablehlo.slice %1809 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1814 = stablehlo.reshape %1813 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1815 = stablehlo.transpose %1814, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %1816 = stablehlo.dot_general %1812, %1815, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %1817 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %1818 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %1819 = stablehlo.select %1817, %1818, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %1820 = stablehlo.broadcast_in_dim %1819, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %1821 = "stablehlo.gather"(%cst_156, %1820) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %1822 = stablehlo.reshape %1821 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %1823 = stablehlo.transpose %1822, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %1824 = stablehlo.broadcast_in_dim %1823, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %1825 = stablehlo.reshape %1824 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %1826 = stablehlo.broadcast_in_dim %1825, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %1827 = stablehlo.add %1816, %1826 : tensor<4x16x49x49xf32>
    %1828 = stablehlo.reduce(%1827 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %1829 = stablehlo.broadcast_in_dim %1828, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %1830 = stablehlo.reshape %1829 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %1831 = stablehlo.broadcast_in_dim %1830, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %1832 = stablehlo.subtract %1827, %1831 : tensor<4x16x49x49xf32>
    %1833 = stablehlo.exponential %1832 : tensor<4x16x49x49xf32>
    %1834 = stablehlo.reduce(%1833 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %1835 = stablehlo.broadcast_in_dim %1834, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %1836 = stablehlo.reshape %1835 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %1837 = stablehlo.broadcast_in_dim %1836, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %1838 = stablehlo.divide %1833, %1837 : tensor<4x16x49x49xf32>
    %1839 = stablehlo.slice %1809 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1840 = stablehlo.reshape %1839 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1841 = stablehlo.dot_general %1838, %1840, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1842 = stablehlo.transpose %1841, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %1843 = stablehlo.reshape %1842 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %1844 = stablehlo.dot_general %1843, %cst_155, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %1845 = stablehlo.reshape %cst_154 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %1846 = stablehlo.broadcast_in_dim %1845, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %1847 = stablehlo.add %1844, %1846 : tensor<4x49x512xf32>
    %1848 = stablehlo.reshape %1847 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %1849 = stablehlo.transpose %1848, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %1850 = stablehlo.reshape %1849 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x14x14x512xf32>
    %1851 = stablehlo.slice %1850 [0:1, 11:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %1852 = stablehlo.slice %1850 [0:1, 0:11, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %1853 = stablehlo.concatenate %1851, %1852, dim = 1 : (tensor<1x3x14x512xf32>, tensor<1x11x14x512xf32>) -> tensor<1x14x14x512xf32>
    %1854 = stablehlo.slice %1853 [0:1, 0:14, 11:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %1855 = stablehlo.slice %1853 [0:1, 0:14, 0:11, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %1856 = stablehlo.concatenate %1854, %1855, dim = 2 : (tensor<1x14x3x512xf32>, tensor<1x14x11x512xf32>) -> tensor<1x14x14x512xf32>
    %1857 = stablehlo.reshape %1856 : (tensor<1x14x14x512xf32>) -> tensor<1x196x512xf32>
    %1858 = stablehlo.add %1769, %1857 : tensor<1x196x512xf32>
    %1859 = stablehlo.reduce(%1858 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1860 = stablehlo.divide %1859, %cst_283 : tensor<1x196xf32>
    %1861 = stablehlo.broadcast_in_dim %1860, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1862 = stablehlo.reshape %1861 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1863 = stablehlo.broadcast_in_dim %1862, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1864 = stablehlo.subtract %1858, %1863 : tensor<1x196x512xf32>
    %1865 = stablehlo.multiply %1858, %1858 : tensor<1x196x512xf32>
    %1866 = stablehlo.reduce(%1865 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1867 = stablehlo.divide %1866, %cst_283 : tensor<1x196xf32>
    %1868 = stablehlo.multiply %1860, %1860 : tensor<1x196xf32>
    %1869 = stablehlo.subtract %1867, %1868 : tensor<1x196xf32>
    %1870 = stablehlo.maximum %cst_288, %1869 : tensor<1x196xf32>
    %1871 = stablehlo.broadcast_in_dim %1870, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1872 = stablehlo.add %1871, %cst_287 : tensor<1x196x1xf32>
    %1873 = stablehlo.rsqrt %1872 : tensor<1x196x1xf32>
    %1874 = stablehlo.reshape %1873 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1875 = stablehlo.broadcast_in_dim %1874, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1876 = stablehlo.reshape %cst_153 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1877 = stablehlo.broadcast_in_dim %1876, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1878 = stablehlo.multiply %1875, %1877 : tensor<1x196x512xf32>
    %1879 = stablehlo.multiply %1864, %1878 : tensor<1x196x512xf32>
    %1880 = stablehlo.reshape %cst_152 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1881 = stablehlo.broadcast_in_dim %1880, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1882 = stablehlo.add %1879, %1881 : tensor<1x196x512xf32>
    %1883 = stablehlo.dot_general %1882, %cst_151, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %1884 = stablehlo.reshape %cst_150 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %1885 = stablehlo.broadcast_in_dim %1884, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %1886 = stablehlo.add %1883, %1885 : tensor<1x196x2048xf32>
    %1887 = stablehlo.multiply %1886, %1886 : tensor<1x196x2048xf32>
    %1888 = stablehlo.multiply %1887, %1886 : tensor<1x196x2048xf32>
    %1889 = stablehlo.multiply %cst_267, %1888 : tensor<1x196x2048xf32>
    %1890 = stablehlo.add %1886, %1889 : tensor<1x196x2048xf32>
    %1891 = stablehlo.multiply %cst_268, %1890 : tensor<1x196x2048xf32>
    %1892 = stablehlo.tanh %1891 : tensor<1x196x2048xf32>
    %1893 = stablehlo.add %cst_269, %1892 : tensor<1x196x2048xf32>
    %1894 = stablehlo.multiply %cst_270, %1893 : tensor<1x196x2048xf32>
    %1895 = stablehlo.multiply %1886, %1894 : tensor<1x196x2048xf32>
    %1896 = stablehlo.dot_general %1895, %cst_149, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %1897 = stablehlo.reshape %cst_148 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1898 = stablehlo.broadcast_in_dim %1897, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1899 = stablehlo.add %1896, %1898 : tensor<1x196x512xf32>
    %1900 = stablehlo.add %1858, %1899 : tensor<1x196x512xf32>
    %1901 = stablehlo.reduce(%1900 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1902 = stablehlo.divide %1901, %cst_283 : tensor<1x196xf32>
    %1903 = stablehlo.broadcast_in_dim %1902, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1904 = stablehlo.reshape %1903 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1905 = stablehlo.broadcast_in_dim %1904, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1906 = stablehlo.subtract %1900, %1905 : tensor<1x196x512xf32>
    %1907 = stablehlo.multiply %1900, %1900 : tensor<1x196x512xf32>
    %1908 = stablehlo.reduce(%1907 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1909 = stablehlo.divide %1908, %cst_283 : tensor<1x196xf32>
    %1910 = stablehlo.multiply %1902, %1902 : tensor<1x196xf32>
    %1911 = stablehlo.subtract %1909, %1910 : tensor<1x196xf32>
    %1912 = stablehlo.maximum %cst_288, %1911 : tensor<1x196xf32>
    %1913 = stablehlo.broadcast_in_dim %1912, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1914 = stablehlo.add %1913, %cst_287 : tensor<1x196x1xf32>
    %1915 = stablehlo.rsqrt %1914 : tensor<1x196x1xf32>
    %1916 = stablehlo.reshape %1915 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1917 = stablehlo.broadcast_in_dim %1916, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1918 = stablehlo.reshape %cst_147 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1919 = stablehlo.broadcast_in_dim %1918, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1920 = stablehlo.multiply %1917, %1919 : tensor<1x196x512xf32>
    %1921 = stablehlo.multiply %1906, %1920 : tensor<1x196x512xf32>
    %1922 = stablehlo.reshape %cst_146 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1923 = stablehlo.broadcast_in_dim %1922, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1924 = stablehlo.add %1921, %1923 : tensor<1x196x512xf32>
    %1925 = stablehlo.reshape %1924 : (tensor<1x196x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %1926 = stablehlo.transpose %1925, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %1927 = stablehlo.reshape %1926 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %1928 = stablehlo.dot_general %1927, %cst_145, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %1929 = stablehlo.reshape %cst_144 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %1930 = stablehlo.broadcast_in_dim %1929, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %1931 = stablehlo.add %1928, %1930 : tensor<4x49x1536xf32>
    %1932 = stablehlo.reshape %1931 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %1933 = stablehlo.transpose %1932, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %1934 = stablehlo.slice %1933 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1935 = stablehlo.reshape %1934 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1936 = stablehlo.multiply %1935, %cst_278 : tensor<4x16x49x32xf32>
    %1937 = stablehlo.slice %1933 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1938 = stablehlo.reshape %1937 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1939 = stablehlo.transpose %1938, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %1940 = stablehlo.dot_general %1936, %1939, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %1941 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %1942 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %1943 = stablehlo.select %1941, %1942, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %1944 = stablehlo.broadcast_in_dim %1943, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %1945 = "stablehlo.gather"(%cst_143, %1944) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %1946 = stablehlo.reshape %1945 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %1947 = stablehlo.transpose %1946, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %1948 = stablehlo.broadcast_in_dim %1947, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %1949 = stablehlo.reshape %1948 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %1950 = stablehlo.broadcast_in_dim %1949, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %1951 = stablehlo.add %1940, %1950 : tensor<4x16x49x49xf32>
    %1952 = stablehlo.reduce(%1951 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %1953 = stablehlo.broadcast_in_dim %1952, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %1954 = stablehlo.reshape %1953 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %1955 = stablehlo.broadcast_in_dim %1954, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %1956 = stablehlo.subtract %1951, %1955 : tensor<4x16x49x49xf32>
    %1957 = stablehlo.exponential %1956 : tensor<4x16x49x49xf32>
    %1958 = stablehlo.reduce(%1957 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %1959 = stablehlo.broadcast_in_dim %1958, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %1960 = stablehlo.reshape %1959 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %1961 = stablehlo.broadcast_in_dim %1960, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %1962 = stablehlo.divide %1957, %1961 : tensor<4x16x49x49xf32>
    %1963 = stablehlo.slice %1933 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %1964 = stablehlo.reshape %1963 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1965 = stablehlo.dot_general %1962, %1964, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %1966 = stablehlo.transpose %1965, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %1967 = stablehlo.reshape %1966 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %1968 = stablehlo.dot_general %1967, %cst_142, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %1969 = stablehlo.reshape %cst_141 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %1970 = stablehlo.broadcast_in_dim %1969, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %1971 = stablehlo.add %1968, %1970 : tensor<4x49x512xf32>
    %1972 = stablehlo.reshape %1971 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %1973 = stablehlo.transpose %1972, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %1974 = stablehlo.reshape %1973 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x196x512xf32>
    %1975 = stablehlo.add %1900, %1974 : tensor<1x196x512xf32>
    %1976 = stablehlo.reduce(%1975 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1977 = stablehlo.divide %1976, %cst_283 : tensor<1x196xf32>
    %1978 = stablehlo.broadcast_in_dim %1977, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1979 = stablehlo.reshape %1978 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1980 = stablehlo.broadcast_in_dim %1979, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1981 = stablehlo.subtract %1975, %1980 : tensor<1x196x512xf32>
    %1982 = stablehlo.multiply %1975, %1975 : tensor<1x196x512xf32>
    %1983 = stablehlo.reduce(%1982 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %1984 = stablehlo.divide %1983, %cst_283 : tensor<1x196xf32>
    %1985 = stablehlo.multiply %1977, %1977 : tensor<1x196xf32>
    %1986 = stablehlo.subtract %1984, %1985 : tensor<1x196xf32>
    %1987 = stablehlo.maximum %cst_288, %1986 : tensor<1x196xf32>
    %1988 = stablehlo.broadcast_in_dim %1987, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %1989 = stablehlo.add %1988, %cst_287 : tensor<1x196x1xf32>
    %1990 = stablehlo.rsqrt %1989 : tensor<1x196x1xf32>
    %1991 = stablehlo.reshape %1990 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %1992 = stablehlo.broadcast_in_dim %1991, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %1993 = stablehlo.reshape %cst_140 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1994 = stablehlo.broadcast_in_dim %1993, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1995 = stablehlo.multiply %1992, %1994 : tensor<1x196x512xf32>
    %1996 = stablehlo.multiply %1981, %1995 : tensor<1x196x512xf32>
    %1997 = stablehlo.reshape %cst_139 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %1998 = stablehlo.broadcast_in_dim %1997, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %1999 = stablehlo.add %1996, %1998 : tensor<1x196x512xf32>
    %2000 = stablehlo.dot_general %1999, %cst_138, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %2001 = stablehlo.reshape %cst_137 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %2002 = stablehlo.broadcast_in_dim %2001, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %2003 = stablehlo.add %2000, %2002 : tensor<1x196x2048xf32>
    %2004 = stablehlo.multiply %2003, %2003 : tensor<1x196x2048xf32>
    %2005 = stablehlo.multiply %2004, %2003 : tensor<1x196x2048xf32>
    %2006 = stablehlo.multiply %cst_267, %2005 : tensor<1x196x2048xf32>
    %2007 = stablehlo.add %2003, %2006 : tensor<1x196x2048xf32>
    %2008 = stablehlo.multiply %cst_268, %2007 : tensor<1x196x2048xf32>
    %2009 = stablehlo.tanh %2008 : tensor<1x196x2048xf32>
    %2010 = stablehlo.add %cst_269, %2009 : tensor<1x196x2048xf32>
    %2011 = stablehlo.multiply %cst_270, %2010 : tensor<1x196x2048xf32>
    %2012 = stablehlo.multiply %2003, %2011 : tensor<1x196x2048xf32>
    %2013 = stablehlo.dot_general %2012, %cst_136, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %2014 = stablehlo.reshape %cst_135 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2015 = stablehlo.broadcast_in_dim %2014, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2016 = stablehlo.add %2013, %2015 : tensor<1x196x512xf32>
    %2017 = stablehlo.add %1975, %2016 : tensor<1x196x512xf32>
    %2018 = stablehlo.reduce(%2017 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2019 = stablehlo.divide %2018, %cst_283 : tensor<1x196xf32>
    %2020 = stablehlo.broadcast_in_dim %2019, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2021 = stablehlo.reshape %2020 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2022 = stablehlo.broadcast_in_dim %2021, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2023 = stablehlo.subtract %2017, %2022 : tensor<1x196x512xf32>
    %2024 = stablehlo.multiply %2017, %2017 : tensor<1x196x512xf32>
    %2025 = stablehlo.reduce(%2024 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2026 = stablehlo.divide %2025, %cst_283 : tensor<1x196xf32>
    %2027 = stablehlo.multiply %2019, %2019 : tensor<1x196xf32>
    %2028 = stablehlo.subtract %2026, %2027 : tensor<1x196xf32>
    %2029 = stablehlo.maximum %cst_288, %2028 : tensor<1x196xf32>
    %2030 = stablehlo.broadcast_in_dim %2029, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2031 = stablehlo.add %2030, %cst_287 : tensor<1x196x1xf32>
    %2032 = stablehlo.rsqrt %2031 : tensor<1x196x1xf32>
    %2033 = stablehlo.reshape %2032 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2034 = stablehlo.broadcast_in_dim %2033, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2035 = stablehlo.reshape %cst_134 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2036 = stablehlo.broadcast_in_dim %2035, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2037 = stablehlo.multiply %2034, %2036 : tensor<1x196x512xf32>
    %2038 = stablehlo.multiply %2023, %2037 : tensor<1x196x512xf32>
    %2039 = stablehlo.reshape %cst_133 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2040 = stablehlo.broadcast_in_dim %2039, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2041 = stablehlo.add %2038, %2040 : tensor<1x196x512xf32>
    %2042 = stablehlo.reshape %2041 : (tensor<1x196x512xf32>) -> tensor<1x14x14x512xf32>
    %2043 = stablehlo.slice %2042 [0:1, 3:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %2044 = stablehlo.slice %2042 [0:1, 0:3, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %2045 = stablehlo.concatenate %2043, %2044, dim = 1 : (tensor<1x11x14x512xf32>, tensor<1x3x14x512xf32>) -> tensor<1x14x14x512xf32>
    %2046 = stablehlo.slice %2045 [0:1, 0:14, 3:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %2047 = stablehlo.slice %2045 [0:1, 0:14, 0:3, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %2048 = stablehlo.concatenate %2046, %2047, dim = 2 : (tensor<1x14x11x512xf32>, tensor<1x14x3x512xf32>) -> tensor<1x14x14x512xf32>
    %2049 = stablehlo.reshape %2048 : (tensor<1x14x14x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %2050 = stablehlo.transpose %2049, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %2051 = stablehlo.reshape %2050 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %2052 = stablehlo.dot_general %2051, %cst_132, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %2053 = stablehlo.reshape %cst_131 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %2054 = stablehlo.broadcast_in_dim %2053, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %2055 = stablehlo.add %2052, %2054 : tensor<4x49x1536xf32>
    %2056 = stablehlo.reshape %2055 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %2057 = stablehlo.transpose %2056, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %2058 = stablehlo.slice %2057 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2059 = stablehlo.reshape %2058 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2060 = stablehlo.multiply %2059, %cst_278 : tensor<4x16x49x32xf32>
    %2061 = stablehlo.slice %2057 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2062 = stablehlo.reshape %2061 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2063 = stablehlo.transpose %2062, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %2064 = stablehlo.dot_general %2060, %2063, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %2065 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %2066 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %2067 = stablehlo.select %2065, %2066, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %2068 = stablehlo.broadcast_in_dim %2067, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %2069 = "stablehlo.gather"(%cst_130, %2068) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %2070 = stablehlo.reshape %2069 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %2071 = stablehlo.transpose %2070, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %2072 = stablehlo.broadcast_in_dim %2071, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %2073 = stablehlo.reshape %2072 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %2074 = stablehlo.broadcast_in_dim %2073, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %2075 = stablehlo.add %2064, %2074 : tensor<4x16x49x49xf32>
    %2076 = stablehlo.reduce(%2075 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %2077 = stablehlo.broadcast_in_dim %2076, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %2078 = stablehlo.reshape %2077 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %2079 = stablehlo.broadcast_in_dim %2078, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %2080 = stablehlo.subtract %2075, %2079 : tensor<4x16x49x49xf32>
    %2081 = stablehlo.exponential %2080 : tensor<4x16x49x49xf32>
    %2082 = stablehlo.reduce(%2081 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %2083 = stablehlo.broadcast_in_dim %2082, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %2084 = stablehlo.reshape %2083 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %2085 = stablehlo.broadcast_in_dim %2084, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %2086 = stablehlo.divide %2081, %2085 : tensor<4x16x49x49xf32>
    %2087 = stablehlo.slice %2057 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2088 = stablehlo.reshape %2087 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2089 = stablehlo.dot_general %2086, %2088, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2090 = stablehlo.transpose %2089, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %2091 = stablehlo.reshape %2090 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %2092 = stablehlo.dot_general %2091, %cst_129, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %2093 = stablehlo.reshape %cst_128 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %2094 = stablehlo.broadcast_in_dim %2093, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %2095 = stablehlo.add %2092, %2094 : tensor<4x49x512xf32>
    %2096 = stablehlo.reshape %2095 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %2097 = stablehlo.transpose %2096, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %2098 = stablehlo.reshape %2097 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x14x14x512xf32>
    %2099 = stablehlo.slice %2098 [0:1, 11:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %2100 = stablehlo.slice %2098 [0:1, 0:11, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %2101 = stablehlo.concatenate %2099, %2100, dim = 1 : (tensor<1x3x14x512xf32>, tensor<1x11x14x512xf32>) -> tensor<1x14x14x512xf32>
    %2102 = stablehlo.slice %2101 [0:1, 0:14, 11:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %2103 = stablehlo.slice %2101 [0:1, 0:14, 0:11, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %2104 = stablehlo.concatenate %2102, %2103, dim = 2 : (tensor<1x14x3x512xf32>, tensor<1x14x11x512xf32>) -> tensor<1x14x14x512xf32>
    %2105 = stablehlo.reshape %2104 : (tensor<1x14x14x512xf32>) -> tensor<1x196x512xf32>
    %2106 = stablehlo.add %2017, %2105 : tensor<1x196x512xf32>
    %2107 = stablehlo.reduce(%2106 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2108 = stablehlo.divide %2107, %cst_283 : tensor<1x196xf32>
    %2109 = stablehlo.broadcast_in_dim %2108, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2110 = stablehlo.reshape %2109 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2111 = stablehlo.broadcast_in_dim %2110, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2112 = stablehlo.subtract %2106, %2111 : tensor<1x196x512xf32>
    %2113 = stablehlo.multiply %2106, %2106 : tensor<1x196x512xf32>
    %2114 = stablehlo.reduce(%2113 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2115 = stablehlo.divide %2114, %cst_283 : tensor<1x196xf32>
    %2116 = stablehlo.multiply %2108, %2108 : tensor<1x196xf32>
    %2117 = stablehlo.subtract %2115, %2116 : tensor<1x196xf32>
    %2118 = stablehlo.maximum %cst_288, %2117 : tensor<1x196xf32>
    %2119 = stablehlo.broadcast_in_dim %2118, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2120 = stablehlo.add %2119, %cst_287 : tensor<1x196x1xf32>
    %2121 = stablehlo.rsqrt %2120 : tensor<1x196x1xf32>
    %2122 = stablehlo.reshape %2121 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2123 = stablehlo.broadcast_in_dim %2122, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2124 = stablehlo.reshape %cst_127 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2125 = stablehlo.broadcast_in_dim %2124, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2126 = stablehlo.multiply %2123, %2125 : tensor<1x196x512xf32>
    %2127 = stablehlo.multiply %2112, %2126 : tensor<1x196x512xf32>
    %2128 = stablehlo.reshape %cst_126 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2129 = stablehlo.broadcast_in_dim %2128, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2130 = stablehlo.add %2127, %2129 : tensor<1x196x512xf32>
    %2131 = stablehlo.dot_general %2130, %cst_125, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %2132 = stablehlo.reshape %cst_124 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %2133 = stablehlo.broadcast_in_dim %2132, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %2134 = stablehlo.add %2131, %2133 : tensor<1x196x2048xf32>
    %2135 = stablehlo.multiply %2134, %2134 : tensor<1x196x2048xf32>
    %2136 = stablehlo.multiply %2135, %2134 : tensor<1x196x2048xf32>
    %2137 = stablehlo.multiply %cst_267, %2136 : tensor<1x196x2048xf32>
    %2138 = stablehlo.add %2134, %2137 : tensor<1x196x2048xf32>
    %2139 = stablehlo.multiply %cst_268, %2138 : tensor<1x196x2048xf32>
    %2140 = stablehlo.tanh %2139 : tensor<1x196x2048xf32>
    %2141 = stablehlo.add %cst_269, %2140 : tensor<1x196x2048xf32>
    %2142 = stablehlo.multiply %cst_270, %2141 : tensor<1x196x2048xf32>
    %2143 = stablehlo.multiply %2134, %2142 : tensor<1x196x2048xf32>
    %2144 = stablehlo.dot_general %2143, %cst_123, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %2145 = stablehlo.reshape %cst_122 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2146 = stablehlo.broadcast_in_dim %2145, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2147 = stablehlo.add %2144, %2146 : tensor<1x196x512xf32>
    %2148 = stablehlo.add %2106, %2147 : tensor<1x196x512xf32>
    %2149 = stablehlo.reduce(%2148 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2150 = stablehlo.divide %2149, %cst_283 : tensor<1x196xf32>
    %2151 = stablehlo.broadcast_in_dim %2150, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2152 = stablehlo.reshape %2151 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2153 = stablehlo.broadcast_in_dim %2152, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2154 = stablehlo.subtract %2148, %2153 : tensor<1x196x512xf32>
    %2155 = stablehlo.multiply %2148, %2148 : tensor<1x196x512xf32>
    %2156 = stablehlo.reduce(%2155 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2157 = stablehlo.divide %2156, %cst_283 : tensor<1x196xf32>
    %2158 = stablehlo.multiply %2150, %2150 : tensor<1x196xf32>
    %2159 = stablehlo.subtract %2157, %2158 : tensor<1x196xf32>
    %2160 = stablehlo.maximum %cst_288, %2159 : tensor<1x196xf32>
    %2161 = stablehlo.broadcast_in_dim %2160, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2162 = stablehlo.add %2161, %cst_287 : tensor<1x196x1xf32>
    %2163 = stablehlo.rsqrt %2162 : tensor<1x196x1xf32>
    %2164 = stablehlo.reshape %2163 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2165 = stablehlo.broadcast_in_dim %2164, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2166 = stablehlo.reshape %cst_121 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2167 = stablehlo.broadcast_in_dim %2166, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2168 = stablehlo.multiply %2165, %2167 : tensor<1x196x512xf32>
    %2169 = stablehlo.multiply %2154, %2168 : tensor<1x196x512xf32>
    %2170 = stablehlo.reshape %cst_120 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2171 = stablehlo.broadcast_in_dim %2170, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2172 = stablehlo.add %2169, %2171 : tensor<1x196x512xf32>
    %2173 = stablehlo.reshape %2172 : (tensor<1x196x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %2174 = stablehlo.transpose %2173, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %2175 = stablehlo.reshape %2174 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %2176 = stablehlo.dot_general %2175, %cst_119, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %2177 = stablehlo.reshape %cst_118 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %2178 = stablehlo.broadcast_in_dim %2177, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %2179 = stablehlo.add %2176, %2178 : tensor<4x49x1536xf32>
    %2180 = stablehlo.reshape %2179 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %2181 = stablehlo.transpose %2180, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %2182 = stablehlo.slice %2181 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2183 = stablehlo.reshape %2182 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2184 = stablehlo.multiply %2183, %cst_278 : tensor<4x16x49x32xf32>
    %2185 = stablehlo.slice %2181 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2186 = stablehlo.reshape %2185 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2187 = stablehlo.transpose %2186, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %2188 = stablehlo.dot_general %2184, %2187, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %2189 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %2190 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %2191 = stablehlo.select %2189, %2190, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %2192 = stablehlo.broadcast_in_dim %2191, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %2193 = "stablehlo.gather"(%cst_117, %2192) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %2194 = stablehlo.reshape %2193 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %2195 = stablehlo.transpose %2194, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %2196 = stablehlo.broadcast_in_dim %2195, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %2197 = stablehlo.reshape %2196 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %2198 = stablehlo.broadcast_in_dim %2197, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %2199 = stablehlo.add %2188, %2198 : tensor<4x16x49x49xf32>
    %2200 = stablehlo.reduce(%2199 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %2201 = stablehlo.broadcast_in_dim %2200, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %2202 = stablehlo.reshape %2201 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %2203 = stablehlo.broadcast_in_dim %2202, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %2204 = stablehlo.subtract %2199, %2203 : tensor<4x16x49x49xf32>
    %2205 = stablehlo.exponential %2204 : tensor<4x16x49x49xf32>
    %2206 = stablehlo.reduce(%2205 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %2207 = stablehlo.broadcast_in_dim %2206, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %2208 = stablehlo.reshape %2207 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %2209 = stablehlo.broadcast_in_dim %2208, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %2210 = stablehlo.divide %2205, %2209 : tensor<4x16x49x49xf32>
    %2211 = stablehlo.slice %2181 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2212 = stablehlo.reshape %2211 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2213 = stablehlo.dot_general %2210, %2212, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2214 = stablehlo.transpose %2213, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %2215 = stablehlo.reshape %2214 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %2216 = stablehlo.dot_general %2215, %cst_116, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %2217 = stablehlo.reshape %cst_115 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %2218 = stablehlo.broadcast_in_dim %2217, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %2219 = stablehlo.add %2216, %2218 : tensor<4x49x512xf32>
    %2220 = stablehlo.reshape %2219 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %2221 = stablehlo.transpose %2220, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %2222 = stablehlo.reshape %2221 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x196x512xf32>
    %2223 = stablehlo.add %2148, %2222 : tensor<1x196x512xf32>
    %2224 = stablehlo.reduce(%2223 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2225 = stablehlo.divide %2224, %cst_283 : tensor<1x196xf32>
    %2226 = stablehlo.broadcast_in_dim %2225, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2227 = stablehlo.reshape %2226 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2228 = stablehlo.broadcast_in_dim %2227, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2229 = stablehlo.subtract %2223, %2228 : tensor<1x196x512xf32>
    %2230 = stablehlo.multiply %2223, %2223 : tensor<1x196x512xf32>
    %2231 = stablehlo.reduce(%2230 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2232 = stablehlo.divide %2231, %cst_283 : tensor<1x196xf32>
    %2233 = stablehlo.multiply %2225, %2225 : tensor<1x196xf32>
    %2234 = stablehlo.subtract %2232, %2233 : tensor<1x196xf32>
    %2235 = stablehlo.maximum %cst_288, %2234 : tensor<1x196xf32>
    %2236 = stablehlo.broadcast_in_dim %2235, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2237 = stablehlo.add %2236, %cst_287 : tensor<1x196x1xf32>
    %2238 = stablehlo.rsqrt %2237 : tensor<1x196x1xf32>
    %2239 = stablehlo.reshape %2238 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2240 = stablehlo.broadcast_in_dim %2239, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2241 = stablehlo.reshape %cst_114 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2242 = stablehlo.broadcast_in_dim %2241, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2243 = stablehlo.multiply %2240, %2242 : tensor<1x196x512xf32>
    %2244 = stablehlo.multiply %2229, %2243 : tensor<1x196x512xf32>
    %2245 = stablehlo.reshape %cst_113 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2246 = stablehlo.broadcast_in_dim %2245, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2247 = stablehlo.add %2244, %2246 : tensor<1x196x512xf32>
    %2248 = stablehlo.dot_general %2247, %cst_112, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %2249 = stablehlo.reshape %cst_111 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %2250 = stablehlo.broadcast_in_dim %2249, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %2251 = stablehlo.add %2248, %2250 : tensor<1x196x2048xf32>
    %2252 = stablehlo.multiply %2251, %2251 : tensor<1x196x2048xf32>
    %2253 = stablehlo.multiply %2252, %2251 : tensor<1x196x2048xf32>
    %2254 = stablehlo.multiply %cst_267, %2253 : tensor<1x196x2048xf32>
    %2255 = stablehlo.add %2251, %2254 : tensor<1x196x2048xf32>
    %2256 = stablehlo.multiply %cst_268, %2255 : tensor<1x196x2048xf32>
    %2257 = stablehlo.tanh %2256 : tensor<1x196x2048xf32>
    %2258 = stablehlo.add %cst_269, %2257 : tensor<1x196x2048xf32>
    %2259 = stablehlo.multiply %cst_270, %2258 : tensor<1x196x2048xf32>
    %2260 = stablehlo.multiply %2251, %2259 : tensor<1x196x2048xf32>
    %2261 = stablehlo.dot_general %2260, %cst_110, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %2262 = stablehlo.reshape %cst_109 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2263 = stablehlo.broadcast_in_dim %2262, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2264 = stablehlo.add %2261, %2263 : tensor<1x196x512xf32>
    %2265 = stablehlo.add %2223, %2264 : tensor<1x196x512xf32>
    %2266 = stablehlo.reduce(%2265 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2267 = stablehlo.divide %2266, %cst_283 : tensor<1x196xf32>
    %2268 = stablehlo.broadcast_in_dim %2267, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2269 = stablehlo.reshape %2268 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2270 = stablehlo.broadcast_in_dim %2269, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2271 = stablehlo.subtract %2265, %2270 : tensor<1x196x512xf32>
    %2272 = stablehlo.multiply %2265, %2265 : tensor<1x196x512xf32>
    %2273 = stablehlo.reduce(%2272 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2274 = stablehlo.divide %2273, %cst_283 : tensor<1x196xf32>
    %2275 = stablehlo.multiply %2267, %2267 : tensor<1x196xf32>
    %2276 = stablehlo.subtract %2274, %2275 : tensor<1x196xf32>
    %2277 = stablehlo.maximum %cst_288, %2276 : tensor<1x196xf32>
    %2278 = stablehlo.broadcast_in_dim %2277, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2279 = stablehlo.add %2278, %cst_287 : tensor<1x196x1xf32>
    %2280 = stablehlo.rsqrt %2279 : tensor<1x196x1xf32>
    %2281 = stablehlo.reshape %2280 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2282 = stablehlo.broadcast_in_dim %2281, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2283 = stablehlo.reshape %cst_108 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2284 = stablehlo.broadcast_in_dim %2283, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2285 = stablehlo.multiply %2282, %2284 : tensor<1x196x512xf32>
    %2286 = stablehlo.multiply %2271, %2285 : tensor<1x196x512xf32>
    %2287 = stablehlo.reshape %cst_107 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2288 = stablehlo.broadcast_in_dim %2287, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2289 = stablehlo.add %2286, %2288 : tensor<1x196x512xf32>
    %2290 = stablehlo.reshape %2289 : (tensor<1x196x512xf32>) -> tensor<1x14x14x512xf32>
    %2291 = stablehlo.slice %2290 [0:1, 3:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %2292 = stablehlo.slice %2290 [0:1, 0:3, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %2293 = stablehlo.concatenate %2291, %2292, dim = 1 : (tensor<1x11x14x512xf32>, tensor<1x3x14x512xf32>) -> tensor<1x14x14x512xf32>
    %2294 = stablehlo.slice %2293 [0:1, 0:14, 3:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %2295 = stablehlo.slice %2293 [0:1, 0:14, 0:3, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %2296 = stablehlo.concatenate %2294, %2295, dim = 2 : (tensor<1x14x11x512xf32>, tensor<1x14x3x512xf32>) -> tensor<1x14x14x512xf32>
    %2297 = stablehlo.reshape %2296 : (tensor<1x14x14x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %2298 = stablehlo.transpose %2297, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %2299 = stablehlo.reshape %2298 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %2300 = stablehlo.dot_general %2299, %cst_106, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %2301 = stablehlo.reshape %cst_105 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %2302 = stablehlo.broadcast_in_dim %2301, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %2303 = stablehlo.add %2300, %2302 : tensor<4x49x1536xf32>
    %2304 = stablehlo.reshape %2303 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %2305 = stablehlo.transpose %2304, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %2306 = stablehlo.slice %2305 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2307 = stablehlo.reshape %2306 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2308 = stablehlo.multiply %2307, %cst_278 : tensor<4x16x49x32xf32>
    %2309 = stablehlo.slice %2305 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2310 = stablehlo.reshape %2309 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2311 = stablehlo.transpose %2310, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %2312 = stablehlo.dot_general %2308, %2311, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %2313 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %2314 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %2315 = stablehlo.select %2313, %2314, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %2316 = stablehlo.broadcast_in_dim %2315, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %2317 = "stablehlo.gather"(%cst_104, %2316) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %2318 = stablehlo.reshape %2317 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %2319 = stablehlo.transpose %2318, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %2320 = stablehlo.broadcast_in_dim %2319, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %2321 = stablehlo.reshape %2320 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %2322 = stablehlo.broadcast_in_dim %2321, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %2323 = stablehlo.add %2312, %2322 : tensor<4x16x49x49xf32>
    %2324 = stablehlo.reduce(%2323 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %2325 = stablehlo.broadcast_in_dim %2324, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %2326 = stablehlo.reshape %2325 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %2327 = stablehlo.broadcast_in_dim %2326, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %2328 = stablehlo.subtract %2323, %2327 : tensor<4x16x49x49xf32>
    %2329 = stablehlo.exponential %2328 : tensor<4x16x49x49xf32>
    %2330 = stablehlo.reduce(%2329 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %2331 = stablehlo.broadcast_in_dim %2330, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %2332 = stablehlo.reshape %2331 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %2333 = stablehlo.broadcast_in_dim %2332, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %2334 = stablehlo.divide %2329, %2333 : tensor<4x16x49x49xf32>
    %2335 = stablehlo.slice %2305 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2336 = stablehlo.reshape %2335 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2337 = stablehlo.dot_general %2334, %2336, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2338 = stablehlo.transpose %2337, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %2339 = stablehlo.reshape %2338 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %2340 = stablehlo.dot_general %2339, %cst_103, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %2341 = stablehlo.reshape %cst_102 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %2342 = stablehlo.broadcast_in_dim %2341, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %2343 = stablehlo.add %2340, %2342 : tensor<4x49x512xf32>
    %2344 = stablehlo.reshape %2343 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %2345 = stablehlo.transpose %2344, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %2346 = stablehlo.reshape %2345 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x14x14x512xf32>
    %2347 = stablehlo.slice %2346 [0:1, 11:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %2348 = stablehlo.slice %2346 [0:1, 0:11, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %2349 = stablehlo.concatenate %2347, %2348, dim = 1 : (tensor<1x3x14x512xf32>, tensor<1x11x14x512xf32>) -> tensor<1x14x14x512xf32>
    %2350 = stablehlo.slice %2349 [0:1, 0:14, 11:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %2351 = stablehlo.slice %2349 [0:1, 0:14, 0:11, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %2352 = stablehlo.concatenate %2350, %2351, dim = 2 : (tensor<1x14x3x512xf32>, tensor<1x14x11x512xf32>) -> tensor<1x14x14x512xf32>
    %2353 = stablehlo.reshape %2352 : (tensor<1x14x14x512xf32>) -> tensor<1x196x512xf32>
    %2354 = stablehlo.add %2265, %2353 : tensor<1x196x512xf32>
    %2355 = stablehlo.reduce(%2354 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2356 = stablehlo.divide %2355, %cst_283 : tensor<1x196xf32>
    %2357 = stablehlo.broadcast_in_dim %2356, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2358 = stablehlo.reshape %2357 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2359 = stablehlo.broadcast_in_dim %2358, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2360 = stablehlo.subtract %2354, %2359 : tensor<1x196x512xf32>
    %2361 = stablehlo.multiply %2354, %2354 : tensor<1x196x512xf32>
    %2362 = stablehlo.reduce(%2361 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2363 = stablehlo.divide %2362, %cst_283 : tensor<1x196xf32>
    %2364 = stablehlo.multiply %2356, %2356 : tensor<1x196xf32>
    %2365 = stablehlo.subtract %2363, %2364 : tensor<1x196xf32>
    %2366 = stablehlo.maximum %cst_288, %2365 : tensor<1x196xf32>
    %2367 = stablehlo.broadcast_in_dim %2366, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2368 = stablehlo.add %2367, %cst_287 : tensor<1x196x1xf32>
    %2369 = stablehlo.rsqrt %2368 : tensor<1x196x1xf32>
    %2370 = stablehlo.reshape %2369 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2371 = stablehlo.broadcast_in_dim %2370, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2372 = stablehlo.reshape %cst_101 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2373 = stablehlo.broadcast_in_dim %2372, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2374 = stablehlo.multiply %2371, %2373 : tensor<1x196x512xf32>
    %2375 = stablehlo.multiply %2360, %2374 : tensor<1x196x512xf32>
    %2376 = stablehlo.reshape %cst_100 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2377 = stablehlo.broadcast_in_dim %2376, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2378 = stablehlo.add %2375, %2377 : tensor<1x196x512xf32>
    %2379 = stablehlo.dot_general %2378, %cst_99, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %2380 = stablehlo.reshape %cst_98 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %2381 = stablehlo.broadcast_in_dim %2380, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %2382 = stablehlo.add %2379, %2381 : tensor<1x196x2048xf32>
    %2383 = stablehlo.multiply %2382, %2382 : tensor<1x196x2048xf32>
    %2384 = stablehlo.multiply %2383, %2382 : tensor<1x196x2048xf32>
    %2385 = stablehlo.multiply %cst_267, %2384 : tensor<1x196x2048xf32>
    %2386 = stablehlo.add %2382, %2385 : tensor<1x196x2048xf32>
    %2387 = stablehlo.multiply %cst_268, %2386 : tensor<1x196x2048xf32>
    %2388 = stablehlo.tanh %2387 : tensor<1x196x2048xf32>
    %2389 = stablehlo.add %cst_269, %2388 : tensor<1x196x2048xf32>
    %2390 = stablehlo.multiply %cst_270, %2389 : tensor<1x196x2048xf32>
    %2391 = stablehlo.multiply %2382, %2390 : tensor<1x196x2048xf32>
    %2392 = stablehlo.dot_general %2391, %cst_97, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %2393 = stablehlo.reshape %cst_96 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2394 = stablehlo.broadcast_in_dim %2393, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2395 = stablehlo.add %2392, %2394 : tensor<1x196x512xf32>
    %2396 = stablehlo.add %2354, %2395 : tensor<1x196x512xf32>
    %2397 = stablehlo.reduce(%2396 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2398 = stablehlo.divide %2397, %cst_283 : tensor<1x196xf32>
    %2399 = stablehlo.broadcast_in_dim %2398, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2400 = stablehlo.reshape %2399 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2401 = stablehlo.broadcast_in_dim %2400, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2402 = stablehlo.subtract %2396, %2401 : tensor<1x196x512xf32>
    %2403 = stablehlo.multiply %2396, %2396 : tensor<1x196x512xf32>
    %2404 = stablehlo.reduce(%2403 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2405 = stablehlo.divide %2404, %cst_283 : tensor<1x196xf32>
    %2406 = stablehlo.multiply %2398, %2398 : tensor<1x196xf32>
    %2407 = stablehlo.subtract %2405, %2406 : tensor<1x196xf32>
    %2408 = stablehlo.maximum %cst_288, %2407 : tensor<1x196xf32>
    %2409 = stablehlo.broadcast_in_dim %2408, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2410 = stablehlo.add %2409, %cst_287 : tensor<1x196x1xf32>
    %2411 = stablehlo.rsqrt %2410 : tensor<1x196x1xf32>
    %2412 = stablehlo.reshape %2411 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2413 = stablehlo.broadcast_in_dim %2412, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2414 = stablehlo.reshape %cst_95 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2415 = stablehlo.broadcast_in_dim %2414, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2416 = stablehlo.multiply %2413, %2415 : tensor<1x196x512xf32>
    %2417 = stablehlo.multiply %2402, %2416 : tensor<1x196x512xf32>
    %2418 = stablehlo.reshape %cst_94 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2419 = stablehlo.broadcast_in_dim %2418, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2420 = stablehlo.add %2417, %2419 : tensor<1x196x512xf32>
    %2421 = stablehlo.reshape %2420 : (tensor<1x196x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %2422 = stablehlo.transpose %2421, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %2423 = stablehlo.reshape %2422 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %2424 = stablehlo.dot_general %2423, %cst_93, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %2425 = stablehlo.reshape %cst_92 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %2426 = stablehlo.broadcast_in_dim %2425, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %2427 = stablehlo.add %2424, %2426 : tensor<4x49x1536xf32>
    %2428 = stablehlo.reshape %2427 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %2429 = stablehlo.transpose %2428, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %2430 = stablehlo.slice %2429 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2431 = stablehlo.reshape %2430 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2432 = stablehlo.multiply %2431, %cst_278 : tensor<4x16x49x32xf32>
    %2433 = stablehlo.slice %2429 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2434 = stablehlo.reshape %2433 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2435 = stablehlo.transpose %2434, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %2436 = stablehlo.dot_general %2432, %2435, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %2437 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %2438 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %2439 = stablehlo.select %2437, %2438, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %2440 = stablehlo.broadcast_in_dim %2439, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %2441 = "stablehlo.gather"(%cst_91, %2440) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %2442 = stablehlo.reshape %2441 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %2443 = stablehlo.transpose %2442, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %2444 = stablehlo.broadcast_in_dim %2443, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %2445 = stablehlo.reshape %2444 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %2446 = stablehlo.broadcast_in_dim %2445, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %2447 = stablehlo.add %2436, %2446 : tensor<4x16x49x49xf32>
    %2448 = stablehlo.reduce(%2447 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %2449 = stablehlo.broadcast_in_dim %2448, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %2450 = stablehlo.reshape %2449 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %2451 = stablehlo.broadcast_in_dim %2450, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %2452 = stablehlo.subtract %2447, %2451 : tensor<4x16x49x49xf32>
    %2453 = stablehlo.exponential %2452 : tensor<4x16x49x49xf32>
    %2454 = stablehlo.reduce(%2453 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %2455 = stablehlo.broadcast_in_dim %2454, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %2456 = stablehlo.reshape %2455 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %2457 = stablehlo.broadcast_in_dim %2456, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %2458 = stablehlo.divide %2453, %2457 : tensor<4x16x49x49xf32>
    %2459 = stablehlo.slice %2429 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2460 = stablehlo.reshape %2459 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2461 = stablehlo.dot_general %2458, %2460, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2462 = stablehlo.transpose %2461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %2463 = stablehlo.reshape %2462 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %2464 = stablehlo.dot_general %2463, %cst_90, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %2465 = stablehlo.reshape %cst_89 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %2466 = stablehlo.broadcast_in_dim %2465, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %2467 = stablehlo.add %2464, %2466 : tensor<4x49x512xf32>
    %2468 = stablehlo.reshape %2467 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %2469 = stablehlo.transpose %2468, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %2470 = stablehlo.reshape %2469 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x196x512xf32>
    %2471 = stablehlo.add %2396, %2470 : tensor<1x196x512xf32>
    %2472 = stablehlo.reduce(%2471 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2473 = stablehlo.divide %2472, %cst_283 : tensor<1x196xf32>
    %2474 = stablehlo.broadcast_in_dim %2473, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2475 = stablehlo.reshape %2474 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2476 = stablehlo.broadcast_in_dim %2475, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2477 = stablehlo.subtract %2471, %2476 : tensor<1x196x512xf32>
    %2478 = stablehlo.multiply %2471, %2471 : tensor<1x196x512xf32>
    %2479 = stablehlo.reduce(%2478 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2480 = stablehlo.divide %2479, %cst_283 : tensor<1x196xf32>
    %2481 = stablehlo.multiply %2473, %2473 : tensor<1x196xf32>
    %2482 = stablehlo.subtract %2480, %2481 : tensor<1x196xf32>
    %2483 = stablehlo.maximum %cst_288, %2482 : tensor<1x196xf32>
    %2484 = stablehlo.broadcast_in_dim %2483, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2485 = stablehlo.add %2484, %cst_287 : tensor<1x196x1xf32>
    %2486 = stablehlo.rsqrt %2485 : tensor<1x196x1xf32>
    %2487 = stablehlo.reshape %2486 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2488 = stablehlo.broadcast_in_dim %2487, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2489 = stablehlo.reshape %cst_88 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2490 = stablehlo.broadcast_in_dim %2489, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2491 = stablehlo.multiply %2488, %2490 : tensor<1x196x512xf32>
    %2492 = stablehlo.multiply %2477, %2491 : tensor<1x196x512xf32>
    %2493 = stablehlo.reshape %cst_87 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2494 = stablehlo.broadcast_in_dim %2493, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2495 = stablehlo.add %2492, %2494 : tensor<1x196x512xf32>
    %2496 = stablehlo.dot_general %2495, %cst_86, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %2497 = stablehlo.reshape %cst_85 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %2498 = stablehlo.broadcast_in_dim %2497, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %2499 = stablehlo.add %2496, %2498 : tensor<1x196x2048xf32>
    %2500 = stablehlo.multiply %2499, %2499 : tensor<1x196x2048xf32>
    %2501 = stablehlo.multiply %2500, %2499 : tensor<1x196x2048xf32>
    %2502 = stablehlo.multiply %cst_267, %2501 : tensor<1x196x2048xf32>
    %2503 = stablehlo.add %2499, %2502 : tensor<1x196x2048xf32>
    %2504 = stablehlo.multiply %cst_268, %2503 : tensor<1x196x2048xf32>
    %2505 = stablehlo.tanh %2504 : tensor<1x196x2048xf32>
    %2506 = stablehlo.add %cst_269, %2505 : tensor<1x196x2048xf32>
    %2507 = stablehlo.multiply %cst_270, %2506 : tensor<1x196x2048xf32>
    %2508 = stablehlo.multiply %2499, %2507 : tensor<1x196x2048xf32>
    %2509 = stablehlo.dot_general %2508, %cst_84, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %2510 = stablehlo.reshape %cst_83 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2511 = stablehlo.broadcast_in_dim %2510, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2512 = stablehlo.add %2509, %2511 : tensor<1x196x512xf32>
    %2513 = stablehlo.add %2471, %2512 : tensor<1x196x512xf32>
    %2514 = stablehlo.reduce(%2513 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2515 = stablehlo.divide %2514, %cst_283 : tensor<1x196xf32>
    %2516 = stablehlo.broadcast_in_dim %2515, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2517 = stablehlo.reshape %2516 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2518 = stablehlo.broadcast_in_dim %2517, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2519 = stablehlo.subtract %2513, %2518 : tensor<1x196x512xf32>
    %2520 = stablehlo.multiply %2513, %2513 : tensor<1x196x512xf32>
    %2521 = stablehlo.reduce(%2520 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2522 = stablehlo.divide %2521, %cst_283 : tensor<1x196xf32>
    %2523 = stablehlo.multiply %2515, %2515 : tensor<1x196xf32>
    %2524 = stablehlo.subtract %2522, %2523 : tensor<1x196xf32>
    %2525 = stablehlo.maximum %cst_288, %2524 : tensor<1x196xf32>
    %2526 = stablehlo.broadcast_in_dim %2525, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2527 = stablehlo.add %2526, %cst_287 : tensor<1x196x1xf32>
    %2528 = stablehlo.rsqrt %2527 : tensor<1x196x1xf32>
    %2529 = stablehlo.reshape %2528 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2530 = stablehlo.broadcast_in_dim %2529, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2531 = stablehlo.reshape %cst_82 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2532 = stablehlo.broadcast_in_dim %2531, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2533 = stablehlo.multiply %2530, %2532 : tensor<1x196x512xf32>
    %2534 = stablehlo.multiply %2519, %2533 : tensor<1x196x512xf32>
    %2535 = stablehlo.reshape %cst_81 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2536 = stablehlo.broadcast_in_dim %2535, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2537 = stablehlo.add %2534, %2536 : tensor<1x196x512xf32>
    %2538 = stablehlo.reshape %2537 : (tensor<1x196x512xf32>) -> tensor<1x14x14x512xf32>
    %2539 = stablehlo.slice %2538 [0:1, 3:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %2540 = stablehlo.slice %2538 [0:1, 0:3, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %2541 = stablehlo.concatenate %2539, %2540, dim = 1 : (tensor<1x11x14x512xf32>, tensor<1x3x14x512xf32>) -> tensor<1x14x14x512xf32>
    %2542 = stablehlo.slice %2541 [0:1, 0:14, 3:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %2543 = stablehlo.slice %2541 [0:1, 0:14, 0:3, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %2544 = stablehlo.concatenate %2542, %2543, dim = 2 : (tensor<1x14x11x512xf32>, tensor<1x14x3x512xf32>) -> tensor<1x14x14x512xf32>
    %2545 = stablehlo.reshape %2544 : (tensor<1x14x14x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %2546 = stablehlo.transpose %2545, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %2547 = stablehlo.reshape %2546 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %2548 = stablehlo.dot_general %2547, %cst_80, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %2549 = stablehlo.reshape %cst_79 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %2550 = stablehlo.broadcast_in_dim %2549, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %2551 = stablehlo.add %2548, %2550 : tensor<4x49x1536xf32>
    %2552 = stablehlo.reshape %2551 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %2553 = stablehlo.transpose %2552, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %2554 = stablehlo.slice %2553 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2555 = stablehlo.reshape %2554 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2556 = stablehlo.multiply %2555, %cst_278 : tensor<4x16x49x32xf32>
    %2557 = stablehlo.slice %2553 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2558 = stablehlo.reshape %2557 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2559 = stablehlo.transpose %2558, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %2560 = stablehlo.dot_general %2556, %2559, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %2561 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %2562 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %2563 = stablehlo.select %2561, %2562, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %2564 = stablehlo.broadcast_in_dim %2563, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %2565 = "stablehlo.gather"(%cst_78, %2564) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %2566 = stablehlo.reshape %2565 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %2567 = stablehlo.transpose %2566, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %2568 = stablehlo.broadcast_in_dim %2567, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %2569 = stablehlo.reshape %2568 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %2570 = stablehlo.broadcast_in_dim %2569, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %2571 = stablehlo.add %2560, %2570 : tensor<4x16x49x49xf32>
    %2572 = stablehlo.reduce(%2571 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %2573 = stablehlo.broadcast_in_dim %2572, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %2574 = stablehlo.reshape %2573 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %2575 = stablehlo.broadcast_in_dim %2574, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %2576 = stablehlo.subtract %2571, %2575 : tensor<4x16x49x49xf32>
    %2577 = stablehlo.exponential %2576 : tensor<4x16x49x49xf32>
    %2578 = stablehlo.reduce(%2577 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %2579 = stablehlo.broadcast_in_dim %2578, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %2580 = stablehlo.reshape %2579 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %2581 = stablehlo.broadcast_in_dim %2580, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %2582 = stablehlo.divide %2577, %2581 : tensor<4x16x49x49xf32>
    %2583 = stablehlo.slice %2553 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2584 = stablehlo.reshape %2583 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2585 = stablehlo.dot_general %2582, %2584, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2586 = stablehlo.transpose %2585, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %2587 = stablehlo.reshape %2586 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %2588 = stablehlo.dot_general %2587, %cst_77, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %2589 = stablehlo.reshape %cst_76 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %2590 = stablehlo.broadcast_in_dim %2589, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %2591 = stablehlo.add %2588, %2590 : tensor<4x49x512xf32>
    %2592 = stablehlo.reshape %2591 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %2593 = stablehlo.transpose %2592, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %2594 = stablehlo.reshape %2593 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x14x14x512xf32>
    %2595 = stablehlo.slice %2594 [0:1, 11:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %2596 = stablehlo.slice %2594 [0:1, 0:11, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %2597 = stablehlo.concatenate %2595, %2596, dim = 1 : (tensor<1x3x14x512xf32>, tensor<1x11x14x512xf32>) -> tensor<1x14x14x512xf32>
    %2598 = stablehlo.slice %2597 [0:1, 0:14, 11:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %2599 = stablehlo.slice %2597 [0:1, 0:14, 0:11, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %2600 = stablehlo.concatenate %2598, %2599, dim = 2 : (tensor<1x14x3x512xf32>, tensor<1x14x11x512xf32>) -> tensor<1x14x14x512xf32>
    %2601 = stablehlo.reshape %2600 : (tensor<1x14x14x512xf32>) -> tensor<1x196x512xf32>
    %2602 = stablehlo.add %2513, %2601 : tensor<1x196x512xf32>
    %2603 = stablehlo.reduce(%2602 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2604 = stablehlo.divide %2603, %cst_283 : tensor<1x196xf32>
    %2605 = stablehlo.broadcast_in_dim %2604, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2606 = stablehlo.reshape %2605 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2607 = stablehlo.broadcast_in_dim %2606, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2608 = stablehlo.subtract %2602, %2607 : tensor<1x196x512xf32>
    %2609 = stablehlo.multiply %2602, %2602 : tensor<1x196x512xf32>
    %2610 = stablehlo.reduce(%2609 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2611 = stablehlo.divide %2610, %cst_283 : tensor<1x196xf32>
    %2612 = stablehlo.multiply %2604, %2604 : tensor<1x196xf32>
    %2613 = stablehlo.subtract %2611, %2612 : tensor<1x196xf32>
    %2614 = stablehlo.maximum %cst_288, %2613 : tensor<1x196xf32>
    %2615 = stablehlo.broadcast_in_dim %2614, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2616 = stablehlo.add %2615, %cst_287 : tensor<1x196x1xf32>
    %2617 = stablehlo.rsqrt %2616 : tensor<1x196x1xf32>
    %2618 = stablehlo.reshape %2617 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2619 = stablehlo.broadcast_in_dim %2618, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2620 = stablehlo.reshape %cst_75 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2621 = stablehlo.broadcast_in_dim %2620, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2622 = stablehlo.multiply %2619, %2621 : tensor<1x196x512xf32>
    %2623 = stablehlo.multiply %2608, %2622 : tensor<1x196x512xf32>
    %2624 = stablehlo.reshape %cst_74 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2625 = stablehlo.broadcast_in_dim %2624, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2626 = stablehlo.add %2623, %2625 : tensor<1x196x512xf32>
    %2627 = stablehlo.dot_general %2626, %cst_73, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %2628 = stablehlo.reshape %cst_72 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %2629 = stablehlo.broadcast_in_dim %2628, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %2630 = stablehlo.add %2627, %2629 : tensor<1x196x2048xf32>
    %2631 = stablehlo.multiply %2630, %2630 : tensor<1x196x2048xf32>
    %2632 = stablehlo.multiply %2631, %2630 : tensor<1x196x2048xf32>
    %2633 = stablehlo.multiply %cst_267, %2632 : tensor<1x196x2048xf32>
    %2634 = stablehlo.add %2630, %2633 : tensor<1x196x2048xf32>
    %2635 = stablehlo.multiply %cst_268, %2634 : tensor<1x196x2048xf32>
    %2636 = stablehlo.tanh %2635 : tensor<1x196x2048xf32>
    %2637 = stablehlo.add %cst_269, %2636 : tensor<1x196x2048xf32>
    %2638 = stablehlo.multiply %cst_270, %2637 : tensor<1x196x2048xf32>
    %2639 = stablehlo.multiply %2630, %2638 : tensor<1x196x2048xf32>
    %2640 = stablehlo.dot_general %2639, %cst_71, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %2641 = stablehlo.reshape %cst_70 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2642 = stablehlo.broadcast_in_dim %2641, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2643 = stablehlo.add %2640, %2642 : tensor<1x196x512xf32>
    %2644 = stablehlo.add %2602, %2643 : tensor<1x196x512xf32>
    %2645 = stablehlo.reduce(%2644 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2646 = stablehlo.divide %2645, %cst_283 : tensor<1x196xf32>
    %2647 = stablehlo.broadcast_in_dim %2646, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2648 = stablehlo.reshape %2647 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2649 = stablehlo.broadcast_in_dim %2648, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2650 = stablehlo.subtract %2644, %2649 : tensor<1x196x512xf32>
    %2651 = stablehlo.multiply %2644, %2644 : tensor<1x196x512xf32>
    %2652 = stablehlo.reduce(%2651 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2653 = stablehlo.divide %2652, %cst_283 : tensor<1x196xf32>
    %2654 = stablehlo.multiply %2646, %2646 : tensor<1x196xf32>
    %2655 = stablehlo.subtract %2653, %2654 : tensor<1x196xf32>
    %2656 = stablehlo.maximum %cst_288, %2655 : tensor<1x196xf32>
    %2657 = stablehlo.broadcast_in_dim %2656, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2658 = stablehlo.add %2657, %cst_287 : tensor<1x196x1xf32>
    %2659 = stablehlo.rsqrt %2658 : tensor<1x196x1xf32>
    %2660 = stablehlo.reshape %2659 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2661 = stablehlo.broadcast_in_dim %2660, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2662 = stablehlo.reshape %cst_69 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2663 = stablehlo.broadcast_in_dim %2662, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2664 = stablehlo.multiply %2661, %2663 : tensor<1x196x512xf32>
    %2665 = stablehlo.multiply %2650, %2664 : tensor<1x196x512xf32>
    %2666 = stablehlo.reshape %cst_68 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2667 = stablehlo.broadcast_in_dim %2666, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2668 = stablehlo.add %2665, %2667 : tensor<1x196x512xf32>
    %2669 = stablehlo.reshape %2668 : (tensor<1x196x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %2670 = stablehlo.transpose %2669, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %2671 = stablehlo.reshape %2670 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %2672 = stablehlo.dot_general %2671, %cst_67, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %2673 = stablehlo.reshape %cst_66 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %2674 = stablehlo.broadcast_in_dim %2673, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %2675 = stablehlo.add %2672, %2674 : tensor<4x49x1536xf32>
    %2676 = stablehlo.reshape %2675 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %2677 = stablehlo.transpose %2676, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %2678 = stablehlo.slice %2677 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2679 = stablehlo.reshape %2678 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2680 = stablehlo.multiply %2679, %cst_278 : tensor<4x16x49x32xf32>
    %2681 = stablehlo.slice %2677 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2682 = stablehlo.reshape %2681 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2683 = stablehlo.transpose %2682, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %2684 = stablehlo.dot_general %2680, %2683, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %2685 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %2686 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %2687 = stablehlo.select %2685, %2686, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %2688 = stablehlo.broadcast_in_dim %2687, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %2689 = "stablehlo.gather"(%cst_65, %2688) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %2690 = stablehlo.reshape %2689 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %2691 = stablehlo.transpose %2690, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %2692 = stablehlo.broadcast_in_dim %2691, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %2693 = stablehlo.reshape %2692 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %2694 = stablehlo.broadcast_in_dim %2693, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %2695 = stablehlo.add %2684, %2694 : tensor<4x16x49x49xf32>
    %2696 = stablehlo.reduce(%2695 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %2697 = stablehlo.broadcast_in_dim %2696, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %2698 = stablehlo.reshape %2697 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %2699 = stablehlo.broadcast_in_dim %2698, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %2700 = stablehlo.subtract %2695, %2699 : tensor<4x16x49x49xf32>
    %2701 = stablehlo.exponential %2700 : tensor<4x16x49x49xf32>
    %2702 = stablehlo.reduce(%2701 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %2703 = stablehlo.broadcast_in_dim %2702, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %2704 = stablehlo.reshape %2703 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %2705 = stablehlo.broadcast_in_dim %2704, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %2706 = stablehlo.divide %2701, %2705 : tensor<4x16x49x49xf32>
    %2707 = stablehlo.slice %2677 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2708 = stablehlo.reshape %2707 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2709 = stablehlo.dot_general %2706, %2708, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2710 = stablehlo.transpose %2709, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %2711 = stablehlo.reshape %2710 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %2712 = stablehlo.dot_general %2711, %cst_64, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %2713 = stablehlo.reshape %cst_63 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %2714 = stablehlo.broadcast_in_dim %2713, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %2715 = stablehlo.add %2712, %2714 : tensor<4x49x512xf32>
    %2716 = stablehlo.reshape %2715 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %2717 = stablehlo.transpose %2716, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %2718 = stablehlo.reshape %2717 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x196x512xf32>
    %2719 = stablehlo.add %2644, %2718 : tensor<1x196x512xf32>
    %2720 = stablehlo.reduce(%2719 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2721 = stablehlo.divide %2720, %cst_283 : tensor<1x196xf32>
    %2722 = stablehlo.broadcast_in_dim %2721, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2723 = stablehlo.reshape %2722 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2724 = stablehlo.broadcast_in_dim %2723, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2725 = stablehlo.subtract %2719, %2724 : tensor<1x196x512xf32>
    %2726 = stablehlo.multiply %2719, %2719 : tensor<1x196x512xf32>
    %2727 = stablehlo.reduce(%2726 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2728 = stablehlo.divide %2727, %cst_283 : tensor<1x196xf32>
    %2729 = stablehlo.multiply %2721, %2721 : tensor<1x196xf32>
    %2730 = stablehlo.subtract %2728, %2729 : tensor<1x196xf32>
    %2731 = stablehlo.maximum %cst_288, %2730 : tensor<1x196xf32>
    %2732 = stablehlo.broadcast_in_dim %2731, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2733 = stablehlo.add %2732, %cst_287 : tensor<1x196x1xf32>
    %2734 = stablehlo.rsqrt %2733 : tensor<1x196x1xf32>
    %2735 = stablehlo.reshape %2734 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2736 = stablehlo.broadcast_in_dim %2735, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2737 = stablehlo.reshape %cst_62 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2738 = stablehlo.broadcast_in_dim %2737, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2739 = stablehlo.multiply %2736, %2738 : tensor<1x196x512xf32>
    %2740 = stablehlo.multiply %2725, %2739 : tensor<1x196x512xf32>
    %2741 = stablehlo.reshape %cst_61 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2742 = stablehlo.broadcast_in_dim %2741, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2743 = stablehlo.add %2740, %2742 : tensor<1x196x512xf32>
    %2744 = stablehlo.dot_general %2743, %cst_60, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %2745 = stablehlo.reshape %cst_59 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %2746 = stablehlo.broadcast_in_dim %2745, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %2747 = stablehlo.add %2744, %2746 : tensor<1x196x2048xf32>
    %2748 = stablehlo.multiply %2747, %2747 : tensor<1x196x2048xf32>
    %2749 = stablehlo.multiply %2748, %2747 : tensor<1x196x2048xf32>
    %2750 = stablehlo.multiply %cst_267, %2749 : tensor<1x196x2048xf32>
    %2751 = stablehlo.add %2747, %2750 : tensor<1x196x2048xf32>
    %2752 = stablehlo.multiply %cst_268, %2751 : tensor<1x196x2048xf32>
    %2753 = stablehlo.tanh %2752 : tensor<1x196x2048xf32>
    %2754 = stablehlo.add %cst_269, %2753 : tensor<1x196x2048xf32>
    %2755 = stablehlo.multiply %cst_270, %2754 : tensor<1x196x2048xf32>
    %2756 = stablehlo.multiply %2747, %2755 : tensor<1x196x2048xf32>
    %2757 = stablehlo.dot_general %2756, %cst_58, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %2758 = stablehlo.reshape %cst_57 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2759 = stablehlo.broadcast_in_dim %2758, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2760 = stablehlo.add %2757, %2759 : tensor<1x196x512xf32>
    %2761 = stablehlo.add %2719, %2760 : tensor<1x196x512xf32>
    %2762 = stablehlo.reduce(%2761 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2763 = stablehlo.divide %2762, %cst_283 : tensor<1x196xf32>
    %2764 = stablehlo.broadcast_in_dim %2763, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2765 = stablehlo.reshape %2764 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2766 = stablehlo.broadcast_in_dim %2765, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2767 = stablehlo.subtract %2761, %2766 : tensor<1x196x512xf32>
    %2768 = stablehlo.multiply %2761, %2761 : tensor<1x196x512xf32>
    %2769 = stablehlo.reduce(%2768 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2770 = stablehlo.divide %2769, %cst_283 : tensor<1x196xf32>
    %2771 = stablehlo.multiply %2763, %2763 : tensor<1x196xf32>
    %2772 = stablehlo.subtract %2770, %2771 : tensor<1x196xf32>
    %2773 = stablehlo.maximum %cst_288, %2772 : tensor<1x196xf32>
    %2774 = stablehlo.broadcast_in_dim %2773, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2775 = stablehlo.add %2774, %cst_287 : tensor<1x196x1xf32>
    %2776 = stablehlo.rsqrt %2775 : tensor<1x196x1xf32>
    %2777 = stablehlo.reshape %2776 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2778 = stablehlo.broadcast_in_dim %2777, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2779 = stablehlo.reshape %cst_56 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2780 = stablehlo.broadcast_in_dim %2779, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2781 = stablehlo.multiply %2778, %2780 : tensor<1x196x512xf32>
    %2782 = stablehlo.multiply %2767, %2781 : tensor<1x196x512xf32>
    %2783 = stablehlo.reshape %cst_55 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2784 = stablehlo.broadcast_in_dim %2783, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2785 = stablehlo.add %2782, %2784 : tensor<1x196x512xf32>
    %2786 = stablehlo.reshape %2785 : (tensor<1x196x512xf32>) -> tensor<1x14x14x512xf32>
    %2787 = stablehlo.slice %2786 [0:1, 3:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %2788 = stablehlo.slice %2786 [0:1, 0:3, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %2789 = stablehlo.concatenate %2787, %2788, dim = 1 : (tensor<1x11x14x512xf32>, tensor<1x3x14x512xf32>) -> tensor<1x14x14x512xf32>
    %2790 = stablehlo.slice %2789 [0:1, 0:14, 3:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %2791 = stablehlo.slice %2789 [0:1, 0:14, 0:3, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %2792 = stablehlo.concatenate %2790, %2791, dim = 2 : (tensor<1x14x11x512xf32>, tensor<1x14x3x512xf32>) -> tensor<1x14x14x512xf32>
    %2793 = stablehlo.reshape %2792 : (tensor<1x14x14x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %2794 = stablehlo.transpose %2793, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,2,7,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %2795 = stablehlo.reshape %2794 : (tensor<1x2x2x7x7x512xf32>) -> tensor<4x49x512xf32>
    %2796 = stablehlo.dot_general %2795, %cst_54, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x1536xf32>) -> tensor<4x49x1536xf32>
    %2797 = stablehlo.reshape %cst_53 : (tensor<1x1x1536xf32>) -> tensor<1536xf32>
    %2798 = stablehlo.broadcast_in_dim %2797, dims = [2] : (tensor<1536xf32>) -> tensor<4x49x1536xf32>
    %2799 = stablehlo.add %2796, %2798 : tensor<4x49x1536xf32>
    %2800 = stablehlo.reshape %2799 : (tensor<4x49x1536xf32>) -> tensor<4x49x3x16x32xf32>
    %2801 = stablehlo.transpose %2800, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,4,16,49,32]{4,2,0,3,1}"} : (tensor<4x49x3x16x32xf32>) -> tensor<3x4x16x49x32xf32>
    %2802 = stablehlo.slice %2801 [0:1, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2803 = stablehlo.reshape %2802 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2804 = stablehlo.multiply %2803, %cst_278 : tensor<4x16x49x32xf32>
    %2805 = stablehlo.slice %2801 [1:2, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2806 = stablehlo.reshape %2805 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2807 = stablehlo.transpose %2806, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[4,16,32,49]{2,3,1,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x16x32x49xf32>
    %2808 = stablehlo.dot_general %2804, %2807, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x32xf32>, tensor<4x16x32x49xf32>) -> tensor<4x16x49x49xf32>
    %2809 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %2810 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %2811 = stablehlo.select %2809, %2810, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %2812 = stablehlo.broadcast_in_dim %2811, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %2813 = "stablehlo.gather"(%cst_52, %2812) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 16>}> : (tensor<169x16xf32>, tensor<2401x1xi32>) -> tensor<2401x16xf32>
    %2814 = stablehlo.reshape %2813 : (tensor<2401x16xf32>) -> tensor<49x49x16xf32>
    %2815 = stablehlo.transpose %2814, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[16,49,49]{0,2,1}"} : (tensor<49x49x16xf32>) -> tensor<16x49x49xf32>
    %2816 = stablehlo.broadcast_in_dim %2815, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<1x16x49x49xf32>
    %2817 = stablehlo.reshape %2816 : (tensor<1x16x49x49xf32>) -> tensor<16x49x49xf32>
    %2818 = stablehlo.broadcast_in_dim %2817, dims = [1, 2, 3] : (tensor<16x49x49xf32>) -> tensor<4x16x49x49xf32>
    %2819 = stablehlo.add %2808, %2818 : tensor<4x16x49x49xf32>
    %2820 = stablehlo.reduce(%2819 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %2821 = stablehlo.broadcast_in_dim %2820, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %2822 = stablehlo.reshape %2821 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %2823 = stablehlo.broadcast_in_dim %2822, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %2824 = stablehlo.subtract %2819, %2823 : tensor<4x16x49x49xf32>
    %2825 = stablehlo.exponential %2824 : tensor<4x16x49x49xf32>
    %2826 = stablehlo.reduce(%2825 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<4x16x49x49xf32>, tensor<f32>) -> tensor<4x16x49xf32>
    %2827 = stablehlo.broadcast_in_dim %2826, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x1xf32>
    %2828 = stablehlo.reshape %2827 : (tensor<4x16x49x1xf32>) -> tensor<4x16x49xf32>
    %2829 = stablehlo.broadcast_in_dim %2828, dims = [0, 1, 2] : (tensor<4x16x49xf32>) -> tensor<4x16x49x49xf32>
    %2830 = stablehlo.divide %2825, %2829 : tensor<4x16x49x49xf32>
    %2831 = stablehlo.slice %2801 [2:3, 0:4, 0:16, 0:49, 0:32] : (tensor<3x4x16x49x32xf32>) -> tensor<1x4x16x49x32xf32>
    %2832 = stablehlo.reshape %2831 : (tensor<1x4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2833 = stablehlo.dot_general %2830, %2832, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x16x49x49xf32>, tensor<4x16x49x32xf32>) -> tensor<4x16x49x32xf32>
    %2834 = stablehlo.transpose %2833, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[4,49,16,32]{3,1,2,0}"} : (tensor<4x16x49x32xf32>) -> tensor<4x49x16x32xf32>
    %2835 = stablehlo.reshape %2834 : (tensor<4x49x16x32xf32>) -> tensor<4x49x512xf32>
    %2836 = stablehlo.dot_general %2835, %cst_51, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x49x512xf32>, tensor<512x512xf32>) -> tensor<4x49x512xf32>
    %2837 = stablehlo.reshape %cst_50 : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %2838 = stablehlo.broadcast_in_dim %2837, dims = [2] : (tensor<512xf32>) -> tensor<4x49x512xf32>
    %2839 = stablehlo.add %2836, %2838 : tensor<4x49x512xf32>
    %2840 = stablehlo.reshape %2839 : (tensor<4x49x512xf32>) -> tensor<1x2x2x7x7x512xf32>
    %2841 = stablehlo.transpose %2840, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,2,7,2,7,512]{5,4,2,3,1,0}"} : (tensor<1x2x2x7x7x512xf32>) -> tensor<1x2x7x2x7x512xf32>
    %2842 = stablehlo.reshape %2841 : (tensor<1x2x7x2x7x512xf32>) -> tensor<1x14x14x512xf32>
    %2843 = stablehlo.slice %2842 [0:1, 11:14, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x3x14x512xf32>
    %2844 = stablehlo.slice %2842 [0:1, 0:11, 0:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x11x14x512xf32>
    %2845 = stablehlo.concatenate %2843, %2844, dim = 1 : (tensor<1x3x14x512xf32>, tensor<1x11x14x512xf32>) -> tensor<1x14x14x512xf32>
    %2846 = stablehlo.slice %2845 [0:1, 0:14, 11:14, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x3x512xf32>
    %2847 = stablehlo.slice %2845 [0:1, 0:14, 0:11, 0:512] : (tensor<1x14x14x512xf32>) -> tensor<1x14x11x512xf32>
    %2848 = stablehlo.concatenate %2846, %2847, dim = 2 : (tensor<1x14x3x512xf32>, tensor<1x14x11x512xf32>) -> tensor<1x14x14x512xf32>
    %2849 = stablehlo.reshape %2848 : (tensor<1x14x14x512xf32>) -> tensor<1x196x512xf32>
    %2850 = stablehlo.add %2761, %2849 : tensor<1x196x512xf32>
    %2851 = stablehlo.reduce(%2850 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2852 = stablehlo.divide %2851, %cst_283 : tensor<1x196xf32>
    %2853 = stablehlo.broadcast_in_dim %2852, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2854 = stablehlo.reshape %2853 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2855 = stablehlo.broadcast_in_dim %2854, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2856 = stablehlo.subtract %2850, %2855 : tensor<1x196x512xf32>
    %2857 = stablehlo.multiply %2850, %2850 : tensor<1x196x512xf32>
    %2858 = stablehlo.reduce(%2857 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x196x512xf32>, tensor<f32>) -> tensor<1x196xf32>
    %2859 = stablehlo.divide %2858, %cst_283 : tensor<1x196xf32>
    %2860 = stablehlo.multiply %2852, %2852 : tensor<1x196xf32>
    %2861 = stablehlo.subtract %2859, %2860 : tensor<1x196xf32>
    %2862 = stablehlo.maximum %cst_288, %2861 : tensor<1x196xf32>
    %2863 = stablehlo.broadcast_in_dim %2862, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x1xf32>
    %2864 = stablehlo.add %2863, %cst_287 : tensor<1x196x1xf32>
    %2865 = stablehlo.rsqrt %2864 : tensor<1x196x1xf32>
    %2866 = stablehlo.reshape %2865 : (tensor<1x196x1xf32>) -> tensor<1x196xf32>
    %2867 = stablehlo.broadcast_in_dim %2866, dims = [0, 1] : (tensor<1x196xf32>) -> tensor<1x196x512xf32>
    %2868 = stablehlo.reshape %cst_49 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2869 = stablehlo.broadcast_in_dim %2868, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2870 = stablehlo.multiply %2867, %2869 : tensor<1x196x512xf32>
    %2871 = stablehlo.multiply %2856, %2870 : tensor<1x196x512xf32>
    %2872 = stablehlo.reshape %cst_48 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2873 = stablehlo.broadcast_in_dim %2872, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2874 = stablehlo.add %2871, %2873 : tensor<1x196x512xf32>
    %2875 = stablehlo.dot_general %2874, %cst_47, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x512xf32>, tensor<512x2048xf32>) -> tensor<1x196x2048xf32>
    %2876 = stablehlo.reshape %cst_46 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %2877 = stablehlo.broadcast_in_dim %2876, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x196x2048xf32>
    %2878 = stablehlo.add %2875, %2877 : tensor<1x196x2048xf32>
    %2879 = stablehlo.multiply %2878, %2878 : tensor<1x196x2048xf32>
    %2880 = stablehlo.multiply %2879, %2878 : tensor<1x196x2048xf32>
    %2881 = stablehlo.multiply %cst_267, %2880 : tensor<1x196x2048xf32>
    %2882 = stablehlo.add %2878, %2881 : tensor<1x196x2048xf32>
    %2883 = stablehlo.multiply %cst_268, %2882 : tensor<1x196x2048xf32>
    %2884 = stablehlo.tanh %2883 : tensor<1x196x2048xf32>
    %2885 = stablehlo.add %cst_269, %2884 : tensor<1x196x2048xf32>
    %2886 = stablehlo.multiply %cst_270, %2885 : tensor<1x196x2048xf32>
    %2887 = stablehlo.multiply %2878, %2886 : tensor<1x196x2048xf32>
    %2888 = stablehlo.dot_general %2887, %cst_45, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x196x2048xf32>, tensor<2048x512xf32>) -> tensor<1x196x512xf32>
    %2889 = stablehlo.reshape %cst_44 : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
    %2890 = stablehlo.broadcast_in_dim %2889, dims = [0, 2] : (tensor<1x512xf32>) -> tensor<1x196x512xf32>
    %2891 = stablehlo.add %2888, %2890 : tensor<1x196x512xf32>
    %2892 = stablehlo.add %2850, %2891 : tensor<1x196x512xf32>
    %2893 = stablehlo.reshape %2892 : (tensor<1x196x512xf32>) -> tensor<1x14x14x512xf32>
    %2894 = stablehlo.iota dim = 0 : tensor<7xi32>
    %2895 = stablehlo.multiply %c_42, %2894 : tensor<7xi32>
    %2896 = stablehlo.add %c_43, %2895 : tensor<7xi32>
    %2897 = stablehlo.broadcast_in_dim %2896, dims = [0] : (tensor<7xi32>) -> tensor<7x7x1xi32>
    %2898 = stablehlo.iota dim = 0 : tensor<7xi32>
    %2899 = stablehlo.multiply %c_42, %2898 : tensor<7xi32>
    %2900 = stablehlo.add %c_43, %2899 : tensor<7xi32>
    %2901 = stablehlo.broadcast_in_dim %2900, dims = [1] : (tensor<7xi32>) -> tensor<7x7x1xi32>
    %2902 = stablehlo.concatenate %2897, %2901, dim = 2 : (tensor<7x7x1xi32>, tensor<7x7x1xi32>) -> tensor<7x7x2xi32>
    %2903 = "stablehlo.gather"(%2893, %2902) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 3], collapsed_slice_dims = [1, 2], start_index_map = [1, 2], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 1, 512>}> : (tensor<1x14x14x512xf32>, tensor<7x7x2xi32>) -> tensor<1x7x7x512xf32>
    %2904 = stablehlo.iota dim = 0 : tensor<7xi32>
    %2905 = stablehlo.multiply %c_42, %2904 : tensor<7xi32>
    %2906 = stablehlo.add %c, %2905 : tensor<7xi32>
    %2907 = stablehlo.broadcast_in_dim %2906, dims = [0] : (tensor<7xi32>) -> tensor<7x7x1xi32>
    %2908 = stablehlo.iota dim = 0 : tensor<7xi32>
    %2909 = stablehlo.multiply %c_42, %2908 : tensor<7xi32>
    %2910 = stablehlo.add %c_43, %2909 : tensor<7xi32>
    %2911 = stablehlo.broadcast_in_dim %2910, dims = [1] : (tensor<7xi32>) -> tensor<7x7x1xi32>
    %2912 = stablehlo.concatenate %2907, %2911, dim = 2 : (tensor<7x7x1xi32>, tensor<7x7x1xi32>) -> tensor<7x7x2xi32>
    %2913 = "stablehlo.gather"(%2893, %2912) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 3], collapsed_slice_dims = [1, 2], start_index_map = [1, 2], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 1, 512>}> : (tensor<1x14x14x512xf32>, tensor<7x7x2xi32>) -> tensor<1x7x7x512xf32>
    %2914 = stablehlo.iota dim = 0 : tensor<7xi32>
    %2915 = stablehlo.multiply %c_42, %2914 : tensor<7xi32>
    %2916 = stablehlo.add %c_43, %2915 : tensor<7xi32>
    %2917 = stablehlo.broadcast_in_dim %2916, dims = [0] : (tensor<7xi32>) -> tensor<7x7x1xi32>
    %2918 = stablehlo.iota dim = 0 : tensor<7xi32>
    %2919 = stablehlo.multiply %c_42, %2918 : tensor<7xi32>
    %2920 = stablehlo.add %c, %2919 : tensor<7xi32>
    %2921 = stablehlo.broadcast_in_dim %2920, dims = [1] : (tensor<7xi32>) -> tensor<7x7x1xi32>
    %2922 = stablehlo.concatenate %2917, %2921, dim = 2 : (tensor<7x7x1xi32>, tensor<7x7x1xi32>) -> tensor<7x7x2xi32>
    %2923 = "stablehlo.gather"(%2893, %2922) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 3], collapsed_slice_dims = [1, 2], start_index_map = [1, 2], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 1, 512>}> : (tensor<1x14x14x512xf32>, tensor<7x7x2xi32>) -> tensor<1x7x7x512xf32>
    %2924 = stablehlo.iota dim = 0 : tensor<7xi32>
    %2925 = stablehlo.multiply %c_42, %2924 : tensor<7xi32>
    %2926 = stablehlo.add %c, %2925 : tensor<7xi32>
    %2927 = stablehlo.broadcast_in_dim %2926, dims = [0] : (tensor<7xi32>) -> tensor<7x7x1xi32>
    %2928 = stablehlo.iota dim = 0 : tensor<7xi32>
    %2929 = stablehlo.multiply %c_42, %2928 : tensor<7xi32>
    %2930 = stablehlo.add %c, %2929 : tensor<7xi32>
    %2931 = stablehlo.broadcast_in_dim %2930, dims = [1] : (tensor<7xi32>) -> tensor<7x7x1xi32>
    %2932 = stablehlo.concatenate %2927, %2931, dim = 2 : (tensor<7x7x1xi32>, tensor<7x7x1xi32>) -> tensor<7x7x2xi32>
    %2933 = "stablehlo.gather"(%2893, %2932) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 3], collapsed_slice_dims = [1, 2], start_index_map = [1, 2], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 1, 512>}> : (tensor<1x14x14x512xf32>, tensor<7x7x2xi32>) -> tensor<1x7x7x512xf32>
    %2934 = stablehlo.concatenate %2903, %2913, %2923, %2933, dim = 3 : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x2048xf32>
    %2935 = stablehlo.reshape %2934 : (tensor<1x7x7x2048xf32>) -> tensor<1x49x2048xf32>
    %2936 = stablehlo.reduce(%2935 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x49x2048xf32>, tensor<f32>) -> tensor<1x49xf32>
    %2937 = stablehlo.divide %2936, %cst_41 : tensor<1x49xf32>
    %2938 = stablehlo.broadcast_in_dim %2937, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1xf32>
    %2939 = stablehlo.reshape %2938 : (tensor<1x49x1xf32>) -> tensor<1x49xf32>
    %2940 = stablehlo.broadcast_in_dim %2939, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x2048xf32>
    %2941 = stablehlo.subtract %2935, %2940 : tensor<1x49x2048xf32>
    %2942 = stablehlo.multiply %2935, %2935 : tensor<1x49x2048xf32>
    %2943 = stablehlo.reduce(%2942 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x49x2048xf32>, tensor<f32>) -> tensor<1x49xf32>
    %2944 = stablehlo.divide %2943, %cst_41 : tensor<1x49xf32>
    %2945 = stablehlo.multiply %2937, %2937 : tensor<1x49xf32>
    %2946 = stablehlo.subtract %2944, %2945 : tensor<1x49xf32>
    %2947 = stablehlo.maximum %cst_40, %2946 : tensor<1x49xf32>
    %2948 = stablehlo.broadcast_in_dim %2947, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1xf32>
    %2949 = stablehlo.add %2948, %cst_39 : tensor<1x49x1xf32>
    %2950 = stablehlo.rsqrt %2949 : tensor<1x49x1xf32>
    %2951 = stablehlo.reshape %2950 : (tensor<1x49x1xf32>) -> tensor<1x49xf32>
    %2952 = stablehlo.broadcast_in_dim %2951, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x2048xf32>
    %2953 = stablehlo.reshape %cst_38 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %2954 = stablehlo.broadcast_in_dim %2953, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x49x2048xf32>
    %2955 = stablehlo.multiply %2952, %2954 : tensor<1x49x2048xf32>
    %2956 = stablehlo.multiply %2941, %2955 : tensor<1x49x2048xf32>
    %2957 = stablehlo.reshape %cst_37 : (tensor<1x1x2048xf32>) -> tensor<1x2048xf32>
    %2958 = stablehlo.broadcast_in_dim %2957, dims = [0, 2] : (tensor<1x2048xf32>) -> tensor<1x49x2048xf32>
    %2959 = stablehlo.add %2956, %2958 : tensor<1x49x2048xf32>
    %2960 = stablehlo.dot_general %2959, %cst_36, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x49x2048xf32>, tensor<2048x1024xf32>) -> tensor<1x49x1024xf32>
    %2961 = stablehlo.reduce(%2960 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x49x1024xf32>, tensor<f32>) -> tensor<1x49xf32>
    %2962 = stablehlo.divide %2961, %cst_35 : tensor<1x49xf32>
    %2963 = stablehlo.broadcast_in_dim %2962, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1xf32>
    %2964 = stablehlo.reshape %2963 : (tensor<1x49x1xf32>) -> tensor<1x49xf32>
    %2965 = stablehlo.broadcast_in_dim %2964, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1024xf32>
    %2966 = stablehlo.subtract %2960, %2965 : tensor<1x49x1024xf32>
    %2967 = stablehlo.multiply %2960, %2960 : tensor<1x49x1024xf32>
    %2968 = stablehlo.reduce(%2967 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x49x1024xf32>, tensor<f32>) -> tensor<1x49xf32>
    %2969 = stablehlo.divide %2968, %cst_35 : tensor<1x49xf32>
    %2970 = stablehlo.multiply %2962, %2962 : tensor<1x49xf32>
    %2971 = stablehlo.subtract %2969, %2970 : tensor<1x49xf32>
    %2972 = stablehlo.maximum %cst_40, %2971 : tensor<1x49xf32>
    %2973 = stablehlo.broadcast_in_dim %2972, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1xf32>
    %2974 = stablehlo.add %2973, %cst_39 : tensor<1x49x1xf32>
    %2975 = stablehlo.rsqrt %2974 : tensor<1x49x1xf32>
    %2976 = stablehlo.reshape %2975 : (tensor<1x49x1xf32>) -> tensor<1x49xf32>
    %2977 = stablehlo.broadcast_in_dim %2976, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1024xf32>
    %2978 = stablehlo.reshape %cst_34 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %2979 = stablehlo.broadcast_in_dim %2978, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x49x1024xf32>
    %2980 = stablehlo.multiply %2977, %2979 : tensor<1x49x1024xf32>
    %2981 = stablehlo.multiply %2966, %2980 : tensor<1x49x1024xf32>
    %2982 = stablehlo.reshape %cst_33 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %2983 = stablehlo.broadcast_in_dim %2982, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x49x1024xf32>
    %2984 = stablehlo.add %2981, %2983 : tensor<1x49x1024xf32>
    %2985 = stablehlo.reshape %2984 : (tensor<1x49x1024xf32>) -> tensor<1x1x7x1x7x1024xf32>
    %2986 = stablehlo.transpose %2985, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,1,1,7,7,1024]{5,4,2,3,1,0}"} : (tensor<1x1x7x1x7x1024xf32>) -> tensor<1x1x1x7x7x1024xf32>
    %2987 = stablehlo.reshape %2986 : (tensor<1x1x1x7x7x1024xf32>) -> tensor<1x49x1024xf32>
    %2988 = stablehlo.dot_general %2987, %cst_32, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x49x1024xf32>, tensor<1024x3072xf32>) -> tensor<1x49x3072xf32>
    %2989 = stablehlo.reshape %cst_31 : (tensor<1x1x3072xf32>) -> tensor<1x3072xf32>
    %2990 = stablehlo.broadcast_in_dim %2989, dims = [0, 2] : (tensor<1x3072xf32>) -> tensor<1x49x3072xf32>
    %2991 = stablehlo.add %2988, %2990 : tensor<1x49x3072xf32>
    %2992 = stablehlo.reshape %2991 : (tensor<1x49x3072xf32>) -> tensor<1x49x3x32x32xf32>
    %2993 = stablehlo.transpose %2992, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,1,32,49,32]{4,2,0,3,1}"} : (tensor<1x49x3x32x32xf32>) -> tensor<3x1x32x49x32xf32>
    %2994 = stablehlo.slice %2993 [0:1, 0:1, 0:32, 0:49, 0:32] : (tensor<3x1x32x49x32xf32>) -> tensor<1x1x32x49x32xf32>
    %2995 = stablehlo.reshape %2994 : (tensor<1x1x32x49x32xf32>) -> tensor<1x32x49x32xf32>
    %2996 = stablehlo.multiply %2995, %cst_30 : tensor<1x32x49x32xf32>
    %2997 = stablehlo.reshape %2996 : (tensor<1x32x49x32xf32>) -> tensor<32x49x32xf32>
    %2998 = stablehlo.slice %2993 [1:2, 0:1, 0:32, 0:49, 0:32] : (tensor<3x1x32x49x32xf32>) -> tensor<1x1x32x49x32xf32>
    %2999 = stablehlo.reshape %2998 : (tensor<1x1x32x49x32xf32>) -> tensor<1x32x49x32xf32>
    %3000 = stablehlo.transpose %2999, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,32,49]{2,3,1,0}"} : (tensor<1x32x49x32xf32>) -> tensor<1x32x32x49xf32>
    %3001 = stablehlo.dot_general %2997, %3000, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x49x32xf32>, tensor<1x32x32x49xf32>) -> tensor<32x49x1x49xf32>
    %3002 = stablehlo.transpose %3001, dims = [2, 0, 1, 3] {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, xla_shape = "f32[1,32,49,49]{3,0,2,1}"} : (tensor<32x49x1x49xf32>) -> tensor<1x32x49x49xf32>
    %3003 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %3004 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %3005 = stablehlo.select %3003, %3004, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %3006 = stablehlo.broadcast_in_dim %3005, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %3007 = "stablehlo.gather"(%cst_29, %3006) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32>}> : (tensor<169x32xf32>, tensor<2401x1xi32>) -> tensor<2401x32xf32>
    %3008 = stablehlo.reshape %3007 : (tensor<2401x32xf32>) -> tensor<49x49x32xf32>
    %3009 = stablehlo.transpose %3008, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[32,49,49]{0,2,1}"} : (tensor<49x49x32xf32>) -> tensor<32x49x49xf32>
    %3010 = stablehlo.broadcast_in_dim %3009, dims = [1, 2, 3] : (tensor<32x49x49xf32>) -> tensor<1x32x49x49xf32>
    %3011 = stablehlo.add %3002, %3010 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, xla_shape = "f32[1,32,49,49]{3,0,2,1}"} : tensor<1x32x49x49xf32>
    %3012 = stablehlo.reduce(%3011 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x49x49xf32>, tensor<f32>) -> tensor<1x32x49xf32>
    %3013 = stablehlo.broadcast_in_dim %3012, dims = [0, 1, 2] : (tensor<1x32x49xf32>) -> tensor<1x32x49x1xf32>
    %3014 = stablehlo.reshape %3013 : (tensor<1x32x49x1xf32>) -> tensor<1x32x49xf32>
    %3015 = stablehlo.broadcast_in_dim %3014, dims = [0, 1, 2] : (tensor<1x32x49xf32>) -> tensor<1x32x49x49xf32>
    %3016 = stablehlo.subtract %3011, %3015 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, xla_shape = "f32[1,32,49,49]{3,0,2,1}"} : tensor<1x32x49x49xf32>
    %3017 = stablehlo.exponential %3016 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, xla_shape = "f32[1,32,49,49]{3,0,2,1}"} : tensor<1x32x49x49xf32>
    %3018 = stablehlo.reduce(%3017 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<1x32x49x49xf32>, tensor<f32>) -> tensor<1x32x49xf32>
    %3019 = stablehlo.broadcast_in_dim %3018, dims = [0, 1, 2] : (tensor<1x32x49xf32>) -> tensor<1x32x49x1xf32>
    %3020 = stablehlo.reshape %3019 : (tensor<1x32x49x1xf32>) -> tensor<1x32x49xf32>
    %3021 = stablehlo.broadcast_in_dim %3020, dims = [0, 1, 2] : (tensor<1x32x49xf32>) -> tensor<1x32x49x49xf32>
    %3022 = stablehlo.divide %3017, %3021 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, xla_shape = "f32[1,32,49,49]{3,0,2,1}"} : tensor<1x32x49x49xf32>
    %3023 = stablehlo.reshape %3022 : (tensor<1x32x49x49xf32>) -> tensor<32x49x49xf32>
    %3024 = stablehlo.slice %2993 [2:3, 0:1, 0:32, 0:49, 0:32] : (tensor<3x1x32x49x32xf32>) -> tensor<1x1x32x49x32xf32>
    %3025 = stablehlo.reshape %3024 : (tensor<1x1x32x49x32xf32>) -> tensor<1x32x49x32xf32>
    %3026 = stablehlo.dot_general %3023, %3025, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x49x49xf32>, tensor<1x32x49x32xf32>) -> tensor<32x49x1x32xf32>
    %3027 = stablehlo.transpose %3026, dims = [2, 0, 1, 3] {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, xla_shape = "f32[1,32,49,32]{3,0,2,1}"} : (tensor<32x49x1x32xf32>) -> tensor<1x32x49x32xf32>
    %3028 = stablehlo.transpose %3027, dims = [0, 2, 1, 3] {result_layout = dense<[3, 0, 1, 2]> : tensor<4xindex>, xla_shape = "f32[1,49,32,32]{3,0,1,2}"} : (tensor<1x32x49x32xf32>) -> tensor<1x49x32x32xf32>
    %3029 = stablehlo.reshape %3028 : (tensor<1x49x32x32xf32>) -> tensor<1x49x1024xf32>
    %3030 = stablehlo.dot_general %3029, %cst_28, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x49x1024xf32>, tensor<1024x1024xf32>) -> tensor<1x49x1024xf32>
    %3031 = stablehlo.reshape %cst_27 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %3032 = stablehlo.broadcast_in_dim %3031, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x49x1024xf32>
    %3033 = stablehlo.add %3030, %3032 : tensor<1x49x1024xf32>
    %3034 = stablehlo.reshape %3033 : (tensor<1x49x1024xf32>) -> tensor<1x1x1x7x7x1024xf32>
    %3035 = stablehlo.transpose %3034, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,1,7,1,7,1024]{5,4,2,3,1,0}"} : (tensor<1x1x1x7x7x1024xf32>) -> tensor<1x1x7x1x7x1024xf32>
    %3036 = stablehlo.reshape %3035 : (tensor<1x1x7x1x7x1024xf32>) -> tensor<1x49x1024xf32>
    %3037 = stablehlo.add %2960, %3036 : tensor<1x49x1024xf32>
    %3038 = stablehlo.reduce(%3037 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x49x1024xf32>, tensor<f32>) -> tensor<1x49xf32>
    %3039 = stablehlo.divide %3038, %cst_35 : tensor<1x49xf32>
    %3040 = stablehlo.broadcast_in_dim %3039, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1xf32>
    %3041 = stablehlo.reshape %3040 : (tensor<1x49x1xf32>) -> tensor<1x49xf32>
    %3042 = stablehlo.broadcast_in_dim %3041, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1024xf32>
    %3043 = stablehlo.subtract %3037, %3042 : tensor<1x49x1024xf32>
    %3044 = stablehlo.multiply %3037, %3037 : tensor<1x49x1024xf32>
    %3045 = stablehlo.reduce(%3044 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x49x1024xf32>, tensor<f32>) -> tensor<1x49xf32>
    %3046 = stablehlo.divide %3045, %cst_35 : tensor<1x49xf32>
    %3047 = stablehlo.multiply %3039, %3039 : tensor<1x49xf32>
    %3048 = stablehlo.subtract %3046, %3047 : tensor<1x49xf32>
    %3049 = stablehlo.maximum %cst_40, %3048 : tensor<1x49xf32>
    %3050 = stablehlo.broadcast_in_dim %3049, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1xf32>
    %3051 = stablehlo.add %3050, %cst_39 : tensor<1x49x1xf32>
    %3052 = stablehlo.rsqrt %3051 : tensor<1x49x1xf32>
    %3053 = stablehlo.reshape %3052 : (tensor<1x49x1xf32>) -> tensor<1x49xf32>
    %3054 = stablehlo.broadcast_in_dim %3053, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1024xf32>
    %3055 = stablehlo.reshape %cst_26 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %3056 = stablehlo.broadcast_in_dim %3055, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x49x1024xf32>
    %3057 = stablehlo.multiply %3054, %3056 : tensor<1x49x1024xf32>
    %3058 = stablehlo.multiply %3043, %3057 : tensor<1x49x1024xf32>
    %3059 = stablehlo.reshape %cst_25 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %3060 = stablehlo.broadcast_in_dim %3059, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x49x1024xf32>
    %3061 = stablehlo.add %3058, %3060 : tensor<1x49x1024xf32>
    %3062 = stablehlo.dot_general %3061, %cst_24, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x49x1024xf32>, tensor<1024x4096xf32>) -> tensor<1x49x4096xf32>
    %3063 = stablehlo.reshape %cst_23 : (tensor<1x1x4096xf32>) -> tensor<1x4096xf32>
    %3064 = stablehlo.broadcast_in_dim %3063, dims = [0, 2] : (tensor<1x4096xf32>) -> tensor<1x49x4096xf32>
    %3065 = stablehlo.add %3062, %3064 : tensor<1x49x4096xf32>
    %3066 = stablehlo.multiply %3065, %3065 : tensor<1x49x4096xf32>
    %3067 = stablehlo.multiply %3066, %3065 : tensor<1x49x4096xf32>
    %3068 = stablehlo.multiply %cst_19, %3067 : tensor<1x49x4096xf32>
    %3069 = stablehlo.add %3065, %3068 : tensor<1x49x4096xf32>
    %3070 = stablehlo.multiply %cst_20, %3069 : tensor<1x49x4096xf32>
    %3071 = stablehlo.tanh %3070 : tensor<1x49x4096xf32>
    %3072 = stablehlo.add %cst_21, %3071 : tensor<1x49x4096xf32>
    %3073 = stablehlo.multiply %cst_22, %3072 : tensor<1x49x4096xf32>
    %3074 = stablehlo.multiply %3065, %3073 : tensor<1x49x4096xf32>
    %3075 = stablehlo.dot_general %3074, %cst_18, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x49x4096xf32>, tensor<4096x1024xf32>) -> tensor<1x49x1024xf32>
    %3076 = stablehlo.reshape %cst_17 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %3077 = stablehlo.broadcast_in_dim %3076, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x49x1024xf32>
    %3078 = stablehlo.add %3075, %3077 : tensor<1x49x1024xf32>
    %3079 = stablehlo.add %3037, %3078 : tensor<1x49x1024xf32>
    %3080 = stablehlo.reduce(%3079 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x49x1024xf32>, tensor<f32>) -> tensor<1x49xf32>
    %3081 = stablehlo.divide %3080, %cst_35 : tensor<1x49xf32>
    %3082 = stablehlo.broadcast_in_dim %3081, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1xf32>
    %3083 = stablehlo.reshape %3082 : (tensor<1x49x1xf32>) -> tensor<1x49xf32>
    %3084 = stablehlo.broadcast_in_dim %3083, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1024xf32>
    %3085 = stablehlo.subtract %3079, %3084 : tensor<1x49x1024xf32>
    %3086 = stablehlo.multiply %3079, %3079 : tensor<1x49x1024xf32>
    %3087 = stablehlo.reduce(%3086 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x49x1024xf32>, tensor<f32>) -> tensor<1x49xf32>
    %3088 = stablehlo.divide %3087, %cst_35 : tensor<1x49xf32>
    %3089 = stablehlo.multiply %3081, %3081 : tensor<1x49xf32>
    %3090 = stablehlo.subtract %3088, %3089 : tensor<1x49xf32>
    %3091 = stablehlo.maximum %cst_40, %3090 : tensor<1x49xf32>
    %3092 = stablehlo.broadcast_in_dim %3091, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1xf32>
    %3093 = stablehlo.add %3092, %cst_39 : tensor<1x49x1xf32>
    %3094 = stablehlo.rsqrt %3093 : tensor<1x49x1xf32>
    %3095 = stablehlo.reshape %3094 : (tensor<1x49x1xf32>) -> tensor<1x49xf32>
    %3096 = stablehlo.broadcast_in_dim %3095, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1024xf32>
    %3097 = stablehlo.reshape %cst_16 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %3098 = stablehlo.broadcast_in_dim %3097, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x49x1024xf32>
    %3099 = stablehlo.multiply %3096, %3098 : tensor<1x49x1024xf32>
    %3100 = stablehlo.multiply %3085, %3099 : tensor<1x49x1024xf32>
    %3101 = stablehlo.reshape %cst_15 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %3102 = stablehlo.broadcast_in_dim %3101, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x49x1024xf32>
    %3103 = stablehlo.add %3100, %3102 : tensor<1x49x1024xf32>
    %3104 = stablehlo.reshape %3103 : (tensor<1x49x1024xf32>) -> tensor<1x7x7x1024xf32>
    %3105 = stablehlo.slice %3104 [0:1, 3:7, 0:7, 0:1024] : (tensor<1x7x7x1024xf32>) -> tensor<1x4x7x1024xf32>
    %3106 = stablehlo.slice %3104 [0:1, 0:3, 0:7, 0:1024] : (tensor<1x7x7x1024xf32>) -> tensor<1x3x7x1024xf32>
    %3107 = stablehlo.concatenate %3105, %3106, dim = 1 : (tensor<1x4x7x1024xf32>, tensor<1x3x7x1024xf32>) -> tensor<1x7x7x1024xf32>
    %3108 = stablehlo.slice %3107 [0:1, 0:7, 3:7, 0:1024] : (tensor<1x7x7x1024xf32>) -> tensor<1x7x4x1024xf32>
    %3109 = stablehlo.slice %3107 [0:1, 0:7, 0:3, 0:1024] : (tensor<1x7x7x1024xf32>) -> tensor<1x7x3x1024xf32>
    %3110 = stablehlo.concatenate %3108, %3109, dim = 2 : (tensor<1x7x4x1024xf32>, tensor<1x7x3x1024xf32>) -> tensor<1x7x7x1024xf32>
    %3111 = stablehlo.reshape %3110 : (tensor<1x7x7x1024xf32>) -> tensor<1x1x7x1x7x1024xf32>
    %3112 = stablehlo.transpose %3111, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,1,1,7,7,1024]{5,4,2,3,1,0}"} : (tensor<1x1x7x1x7x1024xf32>) -> tensor<1x1x1x7x7x1024xf32>
    %3113 = stablehlo.reshape %3112 : (tensor<1x1x1x7x7x1024xf32>) -> tensor<1x49x1024xf32>
    %3114 = stablehlo.dot_general %3113, %cst_14, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x49x1024xf32>, tensor<1024x3072xf32>) -> tensor<1x49x3072xf32>
    %3115 = stablehlo.reshape %cst_13 : (tensor<1x1x3072xf32>) -> tensor<1x3072xf32>
    %3116 = stablehlo.broadcast_in_dim %3115, dims = [0, 2] : (tensor<1x3072xf32>) -> tensor<1x49x3072xf32>
    %3117 = stablehlo.add %3114, %3116 : tensor<1x49x3072xf32>
    %3118 = stablehlo.reshape %3117 : (tensor<1x49x3072xf32>) -> tensor<1x49x3x32x32xf32>
    %3119 = stablehlo.transpose %3118, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,1,32,49,32]{4,2,0,3,1}"} : (tensor<1x49x3x32x32xf32>) -> tensor<3x1x32x49x32xf32>
    %3120 = stablehlo.slice %3119 [0:1, 0:1, 0:32, 0:49, 0:32] : (tensor<3x1x32x49x32xf32>) -> tensor<1x1x32x49x32xf32>
    %3121 = stablehlo.reshape %3120 : (tensor<1x1x32x49x32xf32>) -> tensor<1x32x49x32xf32>
    %3122 = stablehlo.multiply %3121, %cst_30 : tensor<1x32x49x32xf32>
    %3123 = stablehlo.reshape %3122 : (tensor<1x32x49x32xf32>) -> tensor<32x49x32xf32>
    %3124 = stablehlo.slice %3119 [1:2, 0:1, 0:32, 0:49, 0:32] : (tensor<3x1x32x49x32xf32>) -> tensor<1x1x32x49x32xf32>
    %3125 = stablehlo.reshape %3124 : (tensor<1x1x32x49x32xf32>) -> tensor<1x32x49x32xf32>
    %3126 = stablehlo.transpose %3125, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,32,49]{2,3,1,0}"} : (tensor<1x32x49x32xf32>) -> tensor<1x32x32x49xf32>
    %3127 = stablehlo.dot_general %3123, %3126, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x49x32xf32>, tensor<1x32x32x49xf32>) -> tensor<32x49x1x49xf32>
    %3128 = stablehlo.transpose %3127, dims = [2, 0, 1, 3] {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, xla_shape = "f32[1,32,49,49]{3,0,2,1}"} : (tensor<32x49x1x49xf32>) -> tensor<1x32x49x49xf32>
    %3129 = stablehlo.compare  LT, %c_362, %c_361 : (tensor<2401xi32>, tensor<2401xi32>) -> tensor<2401xi1>
    %3130 = stablehlo.add %c_362, %c_360 : tensor<2401xi32>
    %3131 = stablehlo.select %3129, %3130, %c_362 : tensor<2401xi1>, tensor<2401xi32>
    %3132 = stablehlo.broadcast_in_dim %3131, dims = [0] : (tensor<2401xi32>) -> tensor<2401x1xi32>
    %3133 = "stablehlo.gather"(%cst_12, %3132) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32>}> : (tensor<169x32xf32>, tensor<2401x1xi32>) -> tensor<2401x32xf32>
    %3134 = stablehlo.reshape %3133 : (tensor<2401x32xf32>) -> tensor<49x49x32xf32>
    %3135 = stablehlo.transpose %3134, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[32,49,49]{0,2,1}"} : (tensor<49x49x32xf32>) -> tensor<32x49x49xf32>
    %3136 = stablehlo.broadcast_in_dim %3135, dims = [1, 2, 3] : (tensor<32x49x49xf32>) -> tensor<1x32x49x49xf32>
    %3137 = stablehlo.add %3128, %3136 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, xla_shape = "f32[1,32,49,49]{3,0,2,1}"} : tensor<1x32x49x49xf32>
    %3138 = stablehlo.reduce(%3137 init: %cst_359) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x49x49xf32>, tensor<f32>) -> tensor<1x32x49xf32>
    %3139 = stablehlo.broadcast_in_dim %3138, dims = [0, 1, 2] : (tensor<1x32x49xf32>) -> tensor<1x32x49x1xf32>
    %3140 = stablehlo.reshape %3139 : (tensor<1x32x49x1xf32>) -> tensor<1x32x49xf32>
    %3141 = stablehlo.broadcast_in_dim %3140, dims = [0, 1, 2] : (tensor<1x32x49xf32>) -> tensor<1x32x49x49xf32>
    %3142 = stablehlo.subtract %3137, %3141 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, xla_shape = "f32[1,32,49,49]{3,0,2,1}"} : tensor<1x32x49x49xf32>
    %3143 = stablehlo.exponential %3142 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, xla_shape = "f32[1,32,49,49]{3,0,2,1}"} : tensor<1x32x49x49xf32>
    %3144 = stablehlo.reduce(%3143 init: %cst_374) applies stablehlo.add across dimensions = [3] : (tensor<1x32x49x49xf32>, tensor<f32>) -> tensor<1x32x49xf32>
    %3145 = stablehlo.broadcast_in_dim %3144, dims = [0, 1, 2] : (tensor<1x32x49xf32>) -> tensor<1x32x49x1xf32>
    %3146 = stablehlo.reshape %3145 : (tensor<1x32x49x1xf32>) -> tensor<1x32x49xf32>
    %3147 = stablehlo.broadcast_in_dim %3146, dims = [0, 1, 2] : (tensor<1x32x49xf32>) -> tensor<1x32x49x49xf32>
    %3148 = stablehlo.divide %3143, %3147 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, xla_shape = "f32[1,32,49,49]{3,0,2,1}"} : tensor<1x32x49x49xf32>
    %3149 = stablehlo.reshape %3148 : (tensor<1x32x49x49xf32>) -> tensor<32x49x49xf32>
    %3150 = stablehlo.slice %3119 [2:3, 0:1, 0:32, 0:49, 0:32] : (tensor<3x1x32x49x32xf32>) -> tensor<1x1x32x49x32xf32>
    %3151 = stablehlo.reshape %3150 : (tensor<1x1x32x49x32xf32>) -> tensor<1x32x49x32xf32>
    %3152 = stablehlo.dot_general %3149, %3151, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x49x49xf32>, tensor<1x32x49x32xf32>) -> tensor<32x49x1x32xf32>
    %3153 = stablehlo.transpose %3152, dims = [2, 0, 1, 3] {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, xla_shape = "f32[1,32,49,32]{3,0,2,1}"} : (tensor<32x49x1x32xf32>) -> tensor<1x32x49x32xf32>
    %3154 = stablehlo.transpose %3153, dims = [0, 2, 1, 3] {result_layout = dense<[3, 0, 1, 2]> : tensor<4xindex>, xla_shape = "f32[1,49,32,32]{3,0,1,2}"} : (tensor<1x32x49x32xf32>) -> tensor<1x49x32x32xf32>
    %3155 = stablehlo.reshape %3154 : (tensor<1x49x32x32xf32>) -> tensor<1x49x1024xf32>
    %3156 = stablehlo.dot_general %3155, %cst_11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x49x1024xf32>, tensor<1024x1024xf32>) -> tensor<1x49x1024xf32>
    %3157 = stablehlo.reshape %cst_10 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %3158 = stablehlo.broadcast_in_dim %3157, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x49x1024xf32>
    %3159 = stablehlo.add %3156, %3158 : tensor<1x49x1024xf32>
    %3160 = stablehlo.reshape %3159 : (tensor<1x49x1024xf32>) -> tensor<1x1x1x7x7x1024xf32>
    %3161 = stablehlo.transpose %3160, dims = [0, 1, 3, 2, 4, 5] {result_layout = dense<[5, 4, 2, 3, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1,1,7,1,7,1024]{5,4,2,3,1,0}"} : (tensor<1x1x1x7x7x1024xf32>) -> tensor<1x1x7x1x7x1024xf32>
    %3162 = stablehlo.reshape %3161 : (tensor<1x1x7x1x7x1024xf32>) -> tensor<1x7x7x1024xf32>
    %3163 = stablehlo.slice %3162 [0:1, 4:7, 0:7, 0:1024] : (tensor<1x7x7x1024xf32>) -> tensor<1x3x7x1024xf32>
    %3164 = stablehlo.slice %3162 [0:1, 0:4, 0:7, 0:1024] : (tensor<1x7x7x1024xf32>) -> tensor<1x4x7x1024xf32>
    %3165 = stablehlo.concatenate %3163, %3164, dim = 1 : (tensor<1x3x7x1024xf32>, tensor<1x4x7x1024xf32>) -> tensor<1x7x7x1024xf32>
    %3166 = stablehlo.slice %3165 [0:1, 0:7, 4:7, 0:1024] : (tensor<1x7x7x1024xf32>) -> tensor<1x7x3x1024xf32>
    %3167 = stablehlo.slice %3165 [0:1, 0:7, 0:4, 0:1024] : (tensor<1x7x7x1024xf32>) -> tensor<1x7x4x1024xf32>
    %3168 = stablehlo.concatenate %3166, %3167, dim = 2 : (tensor<1x7x3x1024xf32>, tensor<1x7x4x1024xf32>) -> tensor<1x7x7x1024xf32>
    %3169 = stablehlo.reshape %3168 : (tensor<1x7x7x1024xf32>) -> tensor<1x49x1024xf32>
    %3170 = stablehlo.add %3079, %3169 : tensor<1x49x1024xf32>
    %3171 = stablehlo.reduce(%3170 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x49x1024xf32>, tensor<f32>) -> tensor<1x49xf32>
    %3172 = stablehlo.divide %3171, %cst_35 : tensor<1x49xf32>
    %3173 = stablehlo.broadcast_in_dim %3172, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1xf32>
    %3174 = stablehlo.reshape %3173 : (tensor<1x49x1xf32>) -> tensor<1x49xf32>
    %3175 = stablehlo.broadcast_in_dim %3174, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1024xf32>
    %3176 = stablehlo.subtract %3170, %3175 : tensor<1x49x1024xf32>
    %3177 = stablehlo.multiply %3170, %3170 : tensor<1x49x1024xf32>
    %3178 = stablehlo.reduce(%3177 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x49x1024xf32>, tensor<f32>) -> tensor<1x49xf32>
    %3179 = stablehlo.divide %3178, %cst_35 : tensor<1x49xf32>
    %3180 = stablehlo.multiply %3172, %3172 : tensor<1x49xf32>
    %3181 = stablehlo.subtract %3179, %3180 : tensor<1x49xf32>
    %3182 = stablehlo.maximum %cst_40, %3181 : tensor<1x49xf32>
    %3183 = stablehlo.broadcast_in_dim %3182, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1xf32>
    %3184 = stablehlo.add %3183, %cst_39 : tensor<1x49x1xf32>
    %3185 = stablehlo.rsqrt %3184 : tensor<1x49x1xf32>
    %3186 = stablehlo.reshape %3185 : (tensor<1x49x1xf32>) -> tensor<1x49xf32>
    %3187 = stablehlo.broadcast_in_dim %3186, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1024xf32>
    %3188 = stablehlo.reshape %cst_9 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %3189 = stablehlo.broadcast_in_dim %3188, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x49x1024xf32>
    %3190 = stablehlo.multiply %3187, %3189 : tensor<1x49x1024xf32>
    %3191 = stablehlo.multiply %3176, %3190 : tensor<1x49x1024xf32>
    %3192 = stablehlo.reshape %cst_8 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %3193 = stablehlo.broadcast_in_dim %3192, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x49x1024xf32>
    %3194 = stablehlo.add %3191, %3193 : tensor<1x49x1024xf32>
    %3195 = stablehlo.dot_general %3194, %cst_7, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x49x1024xf32>, tensor<1024x4096xf32>) -> tensor<1x49x4096xf32>
    %3196 = stablehlo.reshape %cst_6 : (tensor<1x1x4096xf32>) -> tensor<1x4096xf32>
    %3197 = stablehlo.broadcast_in_dim %3196, dims = [0, 2] : (tensor<1x4096xf32>) -> tensor<1x49x4096xf32>
    %3198 = stablehlo.add %3195, %3197 : tensor<1x49x4096xf32>
    %3199 = stablehlo.multiply %3198, %3198 : tensor<1x49x4096xf32>
    %3200 = stablehlo.multiply %3199, %3198 : tensor<1x49x4096xf32>
    %3201 = stablehlo.multiply %cst_19, %3200 : tensor<1x49x4096xf32>
    %3202 = stablehlo.add %3198, %3201 : tensor<1x49x4096xf32>
    %3203 = stablehlo.multiply %cst_20, %3202 : tensor<1x49x4096xf32>
    %3204 = stablehlo.tanh %3203 : tensor<1x49x4096xf32>
    %3205 = stablehlo.add %cst_21, %3204 : tensor<1x49x4096xf32>
    %3206 = stablehlo.multiply %cst_22, %3205 : tensor<1x49x4096xf32>
    %3207 = stablehlo.multiply %3198, %3206 : tensor<1x49x4096xf32>
    %3208 = stablehlo.dot_general %3207, %cst_5, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x49x4096xf32>, tensor<4096x1024xf32>) -> tensor<1x49x1024xf32>
    %3209 = stablehlo.reshape %cst_4 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %3210 = stablehlo.broadcast_in_dim %3209, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x49x1024xf32>
    %3211 = stablehlo.add %3208, %3210 : tensor<1x49x1024xf32>
    %3212 = stablehlo.add %3170, %3211 : tensor<1x49x1024xf32>
    %3213 = stablehlo.reduce(%3212 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x49x1024xf32>, tensor<f32>) -> tensor<1x49xf32>
    %3214 = stablehlo.divide %3213, %cst_35 : tensor<1x49xf32>
    %3215 = stablehlo.broadcast_in_dim %3214, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1xf32>
    %3216 = stablehlo.reshape %3215 : (tensor<1x49x1xf32>) -> tensor<1x49xf32>
    %3217 = stablehlo.broadcast_in_dim %3216, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1024xf32>
    %3218 = stablehlo.subtract %3212, %3217 : tensor<1x49x1024xf32>
    %3219 = stablehlo.multiply %3212, %3212 : tensor<1x49x1024xf32>
    %3220 = stablehlo.reduce(%3219 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x49x1024xf32>, tensor<f32>) -> tensor<1x49xf32>
    %3221 = stablehlo.divide %3220, %cst_35 : tensor<1x49xf32>
    %3222 = stablehlo.multiply %3214, %3214 : tensor<1x49xf32>
    %3223 = stablehlo.subtract %3221, %3222 : tensor<1x49xf32>
    %3224 = stablehlo.maximum %cst_40, %3223 : tensor<1x49xf32>
    %3225 = stablehlo.broadcast_in_dim %3224, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1xf32>
    %3226 = stablehlo.add %3225, %cst_39 : tensor<1x49x1xf32>
    %3227 = stablehlo.rsqrt %3226 : tensor<1x49x1xf32>
    %3228 = stablehlo.reshape %3227 : (tensor<1x49x1xf32>) -> tensor<1x49xf32>
    %3229 = stablehlo.broadcast_in_dim %3228, dims = [0, 1] : (tensor<1x49xf32>) -> tensor<1x49x1024xf32>
    %3230 = stablehlo.reshape %cst_3 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %3231 = stablehlo.broadcast_in_dim %3230, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x49x1024xf32>
    %3232 = stablehlo.multiply %3229, %3231 : tensor<1x49x1024xf32>
    %3233 = stablehlo.multiply %3218, %3232 : tensor<1x49x1024xf32>
    %3234 = stablehlo.reshape %cst_2 : (tensor<1x1x1024xf32>) -> tensor<1x1024xf32>
    %3235 = stablehlo.broadcast_in_dim %3234, dims = [0, 2] : (tensor<1x1024xf32>) -> tensor<1x49x1024xf32>
    %3236 = stablehlo.add %3233, %3235 : tensor<1x49x1024xf32>
    %3237 = stablehlo.transpose %3236, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,1024,49]{1,2,0}"} : (tensor<1x49x1024xf32>) -> tensor<1x1024x49xf32>
    %3238 = stablehlo.reduce(%3237 init: %cst_374) applies stablehlo.add across dimensions = [2] : (tensor<1x1024x49xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %3239 = stablehlo.divide %3238, %cst_1 : tensor<1x1024xf32>
    %3240 = stablehlo.dot %3239, %cst_0, precision = [DEFAULT, DEFAULT] : (tensor<1x1024xf32>, tensor<1024x1000xf32>) -> tensor<1x1000xf32>
    %3241 = stablehlo.add %3240, %cst : tensor<1x1000xf32>
    return %3241 : tensor<1x1000xf32>
  }
}


