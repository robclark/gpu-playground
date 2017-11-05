/*
 * Example:
 *
 * Render nodes (minimal): Running a compute shader in a window-less
 *                         EGL + GLES 3.1 context.
 *
 * This example shows the minimum code necessary to run an OpenGL (ES) compute
 * shader (aka, a general purpose program on the GPU), on Linux.
 * It uses the DRM render nodes API to gain unprivileged, shared access to the
 * GPU.
 *
 * See <https://en.wikipedia.org/wiki/Direct_Rendering_Manager#Render_nodes> and
 * <https://dri.freedesktop.org/docs/drm/gpu/drm-uapi.html#render-nodes>.
 *
 * Tested on Linux 4.0, Mesa 12.0, Intel GPU (gen7+).
 *
 * Authors:
 *   * Eduardo Lima Mitev <elima@igalia.com>
 *
 * This code is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * version 3, or (at your option) any later version as published by
 * the Free Software Foundation.
 *
 * THIS CODE IS PROVIDED AS-IS, WITHOUT WARRANTY OF ANY KIND, OR POSSIBLE
 * LIABILITY TO THE AUTHORS FOR ANY CLAIM OR DAMAGE.
 */

#define _GNU_SOURCE

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <assert.h>
#include <fcntl.h>
#include <gbm.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static struct {
   const char *shader;
   const char *device;
   struct {
      unsigned x, y, z;
   } num_groups;
   unsigned bo_size;    /* in dwords */
} opts = {
      /* defaults: */
      .device = "/dev/dri/renderD128",
      .num_groups = { 1, 1, 1 },
      .bo_size = 256,
};

#define get_proc(name) do {                           \
      ext.name = (void *)eglGetProcAddress(#name);    \
   } while (0)

static struct {
   PFNGLDISPATCHCOMPUTEPROC glDispatchCompute;
   PFNGLGETPROGRAMRESOURCEINDEXPROC glGetProgramResourceIndex;
} ext;


const char *readfile(int fd)
{
   static char text[64 * 1024];
   int ret = read(fd, text, sizeof(text));
   if (ret < 0) {
      printf("error reading shader: %d\n", ret);
      exit(-1);
   }
   text[ret] = '\0';
   return strdup(text);
}

static void *mem(int dwords, bool initialize)
{
   unsigned *buf = calloc(dwords, sizeof(buf[0]));

   if (initialize) {
      for (int i = 0; i < dwords; i++) {
         buf[i] = i;
      }
   }

   return buf;
}

static void
hexdump_dwords(const void *data, int sizedwords)
{
   uint32_t *buf = (void *) data;
   int i;

   for (i = 0; i < sizedwords; i++) {
      if (!(i % 8))
         printf("\t%08X:   ", (unsigned int) i*4);
      printf(" %08x", buf[i]);
      if ((i % 8) == 7)
         printf("\n");
   }

   if (i % 8)
      printf("\n");
}

static int setup_tex2d(GLint program, const char *name, GLint unit, bool image)
{
   GLuint tex;
   GLint handle;

   handle = glGetUniformLocation(program, name);
   if (handle >= 0) {
      printf("setup %s\n", name);

      glGenTextures(1, &tex);

      glActiveTexture(GL_TEXTURE0 + unit);
      glBindTexture(GL_TEXTURE_2D, tex);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

      if (image) {
         bool initialize = !!strstr(name, "in");
         glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 64, 64);
         glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 64, 64, GL_RED_INTEGER,
               GL_UNSIGNED_INT, mem(64 * 64 * 4, initialize));
         glBindImageTexture(unit, tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
      } else {
         glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, 64, 64, 0, GL_RED_INTEGER,
               GL_UNSIGNED_INT, mem(64 * 64 * 4, true));
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_LOD, 1);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LOD, 4);

         glUniform1i(handle, unit);
      }

      unit++;
   }

   return unit;
}

static void setup_ubo(GLint program, const char *name)
{
   GLuint ubo = 0, idx;
   static int cnt = 0;
   int i = cnt++;

   idx = glGetUniformBlockIndex(program, name);
   if (idx == GL_INVALID_INDEX)
      return;

   printf("UBO: %s at %u\n", name, idx);

   int sz = opts.bo_size;

   glGenBuffers(1, &ubo);
   glBindBuffer(GL_UNIFORM_BUFFER, ubo);
   glBufferData(GL_UNIFORM_BUFFER, 4 * sz, mem(sz, true), GL_DYNAMIC_DRAW);
   glBindBuffer(GL_UNIFORM_BUFFER, 0);

   glBindBufferBase(GL_UNIFORM_BUFFER, i, ubo);
   glUniformBlockBinding(program, idx, i);
}

static GLuint ssbo_table[256];

static void setup_ssbo(GLint program, const char *name, bool input)
{
   GLuint ssbo = 0, idx;

   idx = ext.glGetProgramResourceIndex(program, GL_SHADER_STORAGE_BLOCK, name);
   if (idx == GL_INVALID_INDEX)
      return;

   printf("SSBO: %s at %u\n", name, idx);

   int sz = opts.bo_size;

   glGenBuffers(1, &ssbo);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
   glBufferData(GL_SHADER_STORAGE_BUFFER, 4 * sz, mem(sz, input), GL_STATIC_DRAW);

   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, idx, ssbo);

   ssbo_table[idx] = ssbo;
}

static void dump_ssbo(GLint program, const char *name)
{
   GLuint idx;
   void *p;

   idx = ext.glGetProgramResourceIndex(program, GL_SHADER_STORAGE_BLOCK, name);
   if (idx == GL_INVALID_INDEX)
      return;

   printf("Dump SSBO: %s at %u\n", name, idx);

   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_table[idx]);
   p = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, opts.bo_size * 4, GL_MAP_READ_BIT);

   hexdump_dwords(p, opts.bo_size);

   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

static void run(void)
{
   bool res;

   int32_t fd = open(opts.device, O_RDWR);
   assert(fd > 0);

   struct gbm_device *gbm = gbm_create_device(fd);
   assert(gbm != NULL);

   get_proc(glDispatchCompute);
   get_proc(glGetProgramResourceIndex);

   /* setup EGL from the GBM device */
   EGLDisplay egl_dpy = eglGetPlatformDisplay(EGL_PLATFORM_GBM_MESA, gbm, NULL);
   assert(egl_dpy != NULL);

   res = eglInitialize(egl_dpy, NULL, NULL);
   assert(res);

   const char *egl_extension_st = eglQueryString(egl_dpy, EGL_EXTENSIONS);
   assert(strstr(egl_extension_st, "EGL_KHR_create_context") != NULL);
   assert(strstr(egl_extension_st, "EGL_KHR_surfaceless_context") != NULL);

   static const EGLint config_attribs[] = {
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
      EGL_NONE
   };
   EGLConfig cfg;
   EGLint count;

   res = eglChooseConfig(egl_dpy, config_attribs, &cfg, 1, &count);
   assert(res);

   res = eglBindAPI(EGL_OPENGL_ES_API);
   assert(res);

   static const EGLint attribs[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3,
      EGL_NONE
   };
   EGLContext core_ctx =
         eglCreateContext(egl_dpy, cfg, EGL_NO_CONTEXT, attribs);
   assert(core_ctx != EGL_NO_CONTEXT);

   res = eglMakeCurrent(egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, core_ctx);
   assert(res);

   /* print some compute limits (not strictly necessary) */
   GLint work_group_count[3] = {0};
   for (unsigned i = 0; i < 3; i++)
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,
                       i,
                       &work_group_count[i]);
   printf("GL_MAX_COMPUTE_WORK_GROUP_COUNT: %d, %d, %d\n",
           work_group_count[0],
           work_group_count[1],
           work_group_count[2]);

   GLint work_group_size[3] = {0};
   for (unsigned i = 0; i < 3; i++)
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, i, &work_group_size[i]);
   printf("GL_MAX_COMPUTE_WORK_GROUP_SIZE: %d, %d, %d\n",
           work_group_size[0],
           work_group_size[1],
           work_group_size[2]);

   GLint max_invocations;
   glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &max_invocations);
   printf("GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS: %d\n", max_invocations);

   GLint mem_size;
   glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &mem_size);
   printf("GL_MAX_COMPUTE_SHARED_MEMORY_SIZE: %d\n", mem_size);

   printf("shader:\n%s\n", opts.shader);

   /* setup a compute shader */
   GLuint compute_shader = glCreateShader(GL_COMPUTE_SHADER);

   assert(glGetError() == GL_NO_ERROR);

   glShaderSource(compute_shader, 1, &opts.shader, NULL);
   assert(glGetError() == GL_NO_ERROR);

   glCompileShader(compute_shader);

   GLint ret;
   glGetShaderiv(compute_shader, GL_COMPILE_STATUS, &ret);
   if (!ret) {
      char *log;

      printf("shader compilation failed!:");
      glGetShaderiv(compute_shader, GL_INFO_LOG_LENGTH, &ret);

      if (ret > 1) {
         log = malloc(ret);
         glGetShaderInfoLog(compute_shader, ret, NULL, log);
         printf("%s", log);
      }
      return;
   }

   GLuint shader_program = glCreateProgram();

   glAttachShader(shader_program, compute_shader);
   assert(glGetError() == GL_NO_ERROR);

   glLinkProgram(shader_program);
   assert(glGetError() == GL_NO_ERROR);

   glDeleteShader(compute_shader);

   glUseProgram(shader_program);
   assert(glGetError() == GL_NO_ERROR);

   setup_ssbo(shader_program, "Input", true);
   setup_ssbo(shader_program, "Output", false);
   setup_ubo(shader_program, "Input");

   GLint unit = 0;
   unit = setup_tex2d(shader_program, "tex2d0", unit, false);
   unit = setup_tex2d(shader_program, "img2d0in", unit, true);
   unit = setup_tex2d(shader_program, "img2d0out", unit, true);

   /* dispatch computation */
   ext.glDispatchCompute(opts.num_groups.x, opts.num_groups.y, opts.num_groups.z);
   assert(glGetError() == GL_NO_ERROR);

   printf("Compute shader dispatched and finished successfully\n");

   dump_ssbo(shader_program, "Output");

   /* free stuff */
   glDeleteProgram(shader_program);
   eglDestroyContext(egl_dpy, core_ctx);
   eglTerminate(egl_dpy);
   gbm_device_destroy(gbm);
   close(fd);
}

static const char *shortopts = "D:G:S:";

static const struct option longopts[] = {
      {"device", required_argument, 0, 'D'},
      {"groups", required_argument, 0, 'G'},
      {"size",   required_argument, 0, 'S'},
      {0, 0, 0, 0}
};

static void usage(const char *name)
{
   printf("Usage: %s [-DGS] SHADER\n"
         "\n"
         "options:\n"
         "    -D, --device=DEVICE      use the given device\n"
         "    -G, --groups=X,Y,Z       use specified group size\n"
         "    -S, --size=DWORDS        size in dwords for UBOs, SSBOs\n"
         ,
         name);
}

int main(int argc, char *argv[])
{
   int opt, ret;

   while ((opt = getopt_long_only(argc, argv, shortopts, longopts, NULL)) != -1) {
      switch (opt) {
      case 'D':
         opts.device = optarg;
         break;
      case 'G':
         ret = sscanf(optarg, "%u,%u,%u", &opts.num_groups.x,
               &opts.num_groups.y, &opts.num_groups.z);
         if (ret != 3)
            goto usage;
         break;
      case 'S':
         opts.bo_size = atoi(optarg);
         break;
      default:
         goto usage;
      }
   }

   if (optind != (argc - 1))
      goto usage;

   opts.shader = readfile(open(argv[optind], 0));

   run();

   return 0;

usage:
   usage(argv[0]);
   return -1;
}
