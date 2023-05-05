using DataFrames, DataFramesMeta
using BenchmarkTools

function fun1(df::DataFrame, bool_exp::BitVector)
    df_temp = df[bool_exp,:]
    return df_temp
end

function fun2(df::DataFrame, bool_exp::BitVector)
    df_temp = @subset(df, identity(bool_exp))
    return df_temp
end

function fun3!(df::DataFrame, bool_exp::BitVector)
    @subset!(df, identity(bool_exp))
    return nothing
end

df = DataFrame(x=rand(10000000),y=rand(10000000),z=rand(10000000))

bool_x = df.x .> 0.5

@benchmark df1=fun1(x, bool_x) setup=(x=copy(df)) evals=1
@benchmark df2=fun2(x, bool_x) setup=(x=copy(df)) evals=1
@benchmark fun3!(x, bool_x) setup=(x=copy(df)) evals=1
# %%

df = DataFrame(x=rand(100),y=rand(100),z=rand(100))
df2 = DataFrame(x=rand(100),y=rand(100),z=rand(100))

function bar!(df,df2)
    bool_x = df.x .> 0.5
    df = df[bool_x,:]
    df2 = df2[bool_x,:]
    df.w = rand(nrow(df))
    df2.w = rand(nrow(df2))
    println(nrow(df))
    println(nrow(df2))
end
bar!(df,df2)
df
df2
