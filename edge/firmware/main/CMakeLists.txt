idf_component_register(SRCS "main.c"
                       INCLUDE_DIRS "." "../.." "../../edge/model"
                       REQUIRES tflite-micro)
