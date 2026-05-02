#version 450

#define PI 3.14159265f

#include "../utility/geometry.glslh"

layout(location = 0) in vec2 inUV;
layout(location = 0) out float outAO;

layout(set = 0, binding = 0) uniform CameraUbo {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 invView;
    mat4 invProj;
    vec3 cameraPosition;
    float padding;
} cam;

layout(set = 0, binding = 1) uniform sampler2D depthTex;
layout(set = 0, binding = 2) uniform sampler2D normalTex;
layout(set = 0, binding = 3) uniform sampler2D directionTex;

layout(set = 0, binding = 4) uniform AOParam {
    vec4  sphereRadius; // r, r^2, 1/r, 0
    int   numSteps;
    int   numDirs;
    float tanAngleBias;
    float aoScalar;
} aoInfo;

vec3 fetchPos(ivec2 iuv)
{
    float depth = texelFetch(depthTex, iuv, 0).r;
    ivec2 sz    = textureSize(depthTex, 0);
    vec2  uv    = (vec2(iuv) + 0.5) / vec2(sz);
    return depthToViewPosition(depth, uv, cam.invProj);
}

vec3 fetchPosFloat(vec2 uv)
{
    float depth = texture(depthTex, uv).r;
    return depthToViewPosition(depth, uv, cam.invProj);
}

vec3 fetchNormal(ivec2 iuv)
{
    return unpackViewNormal(texelFetch(normalTex, iuv, 0).rg);
}

float distSq(vec3 a, vec3 b)
{
    vec3 d = b - a;
    return dot(d, d);
}

float tangent(vec3 p)
{
    return p.z / length(p.xy);
}

float tan2Sin(float x)
{
    return x / sqrt(1.0 + x * x);
}

vec2 rotateDir(vec2 dir, vec2 cosSin)
{
    return vec2(dir.x * cosSin.x - dir.y * cosSin.y,
                dir.x * cosSin.y + dir.y * cosSin.x);
}

vec3 getPointOnPlane(vec4 plane, vec3 pos)
{
    float t = dot(plane.xyz, pos);
    if (t < 0.0)
        pos *= (-plane.w) / t;
    return pos;
}

void computedudv(vec4 plane, vec3 center, inout vec3 du, inout vec3 dv)
{
    ivec2 iuv = ivec2(gl_FragCoord.xy);
    vec3 pt[4];
    pt[0] = fetchPos(iuv + ivec2(-1,  0));
    pt[1] = fetchPos(iuv + ivec2( 1,  0));
    pt[2] = fetchPos(iuv + ivec2( 0, -1));
    pt[3] = fetchPos(iuv + ivec2( 0,  1));

    pt[0] = center - getPointOnPlane(plane, pt[0]);
    pt[1] = getPointOnPlane(plane, pt[1]) - center;
    pt[2] = center - getPointOnPlane(plane, pt[2]);
    pt[3] = getPointOnPlane(plane, pt[3]) - center;

    du = dot(pt[0], pt[0]) < dot(pt[1], pt[1]) ? pt[0] : pt[1];
    dv = dot(pt[2], pt[2]) < dot(pt[3], pt[3]) ? pt[2] : pt[3];

    ivec2 sz = textureSize(depthTex, 0);
    dv *= float(sz.y) / float(sz.x);
}

vec3 tangentVec(vec2 delta, vec3 du, vec3 dv)
{
    return du * delta.x + dv * delta.y;
}

float singleDirectionAO(vec2 deltaUV, vec3 center, vec3 du, vec3 dv, float steps, float randStart, float tanBias)
{
    vec2  uv     = inUV + deltaUV * (1.0 + randStart);
    vec3  tanVec = tangentVec(uv - inUV, du, dv);
    float tanAngle = tangent(tanVec) + tanBias;

    float AO = 0.0;
    float h0 = 0.0;

    for (float i = 0.0; i < steps; i += 1.0) {
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))))
            break;

        vec3  curPos = fetchPosFloat(uv);
        float dsq    = distSq(curPos, center);

        if (dsq < aoInfo.sphereRadius.y) {
            float curTan = tangent(curPos - center);
            if (curTan > tanAngle) {
                float planeTan  = tangent(tangentVec(uv - inUV, du, dv)) + tanBias;
                float curSin    = tan2Sin(curTan);
                float planeSin  = tan2Sin(planeTan);
                float rsq       = dsq * aoInfo.sphereRadius.z * aoInfo.sphereRadius.z;
                float h         = curSin - planeSin;
                AO += (1.0 - rsq) * (h - h0);
                h0       = h;
                tanAngle = curTan;
            }
        }
        uv += deltaUV;
    }
    return AO;
}

float computeAO(vec3 center, vec3 du, vec3 dv, vec2 stepSize, float steps, float tanBias)
{
    float AO         = 0.0;
    float deltaAngle = 2.0 * PI / float(aoInfo.numDirs);

    for (int d = 0; d < aoInfo.numDirs; d++) {
        vec4  jitter  = texelFetch(directionTex, ivec2(d, 0), 0);
        float angle   = deltaAngle * float(d);
        vec2  dir     = vec2(cos(angle), sin(angle));
        vec2  deltaUV = rotateDir(dir, jitter.xy) * stepSize;
        AO += singleDirectionAO(deltaUV, center, du, dv, steps, jitter.z, tanBias);
    }

    return clamp(1.0 - AO / float(aoInfo.numDirs) * aoInfo.aoScalar, 0.0, 1.0);
}

void main()
{
    float depth = texture(depthTex, inUV).r;
    if (depth >= 1.0) {
        outAO = 1.0;
        return;
    }

    vec3 viewPos = depthToViewPosition(depth, inUV, cam.invProj);

    ivec2 sz       = textureSize(depthTex, 0);
    vec2  focalLen = vec2(cam.proj[0][0], cam.proj[1][1]);
    vec2  stepSize = abs(focalLen * 0.5 * aoInfo.sphereRadius.x / viewPos.z);

    float steps = min(float(aoInfo.numSteps),
                  min(stepSize.x * float(sz.x),
                      stepSize.y * float(sz.y)));

    if (steps < 1.0) {
        outAO = 1.0;
        return;
    }

    stepSize /= (steps + 1.0);

    ivec2 iuv       = ivec2(gl_FragCoord.xy);
    vec3  viewNormal = fetchNormal(iuv);

    vec4 plane;
    plane.xyz = viewNormal;
    plane.w   = -dot(viewNormal, viewPos);

    vec3 du, dv;
    computedudv(plane, viewPos, du, dv);

    if (dot(du, du) < 1e-10 || dot(dv, dv) < 1e-10) {
        outAO = 1.0;
        return;
    }

    float viewDotN = abs(dot(normalize(-viewPos), viewNormal));
    float tanBias  = aoInfo.tanAngleBias / max(viewDotN, 0.1);

    outAO = computeAO(viewPos, du, dv, stepSize, steps, tanBias);
}
