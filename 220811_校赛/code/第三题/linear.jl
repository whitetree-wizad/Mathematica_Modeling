### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 5928cfb0-1923-11ed-3ace-2bfad8956c7c
using JuMP,Gurobi,DataFrames,CSV,Parsers,Tables,PlutoUI

# ╔═╡ 274a1543-f320-4705-8e98-55d6a3fbea74
function Base.filter((title,name)::Tuple, df::DataFrame)
	return filter(title=>x -> x == name, df)
end

# ╔═╡ dda52e10-674b-47ab-b765-aa4acf90780a
function Base.filter((title,name)::Tuple, df::DataFrame,week::Integer)
	return filter(title=>x -> x == name, df)[:,"W"*string(week-1)]
end

# ╔═╡ eb6d766a-bc18-4f21-9945-d2bb387a5e6d
function Base.filter((title,name)::Tuple, df::DataFrame,week::String)
	return filter(title=>x -> x == name, df)[:,week]
end

# ╔═╡ d2e2fc85-3b17-4fce-af9b-5f1d38109691
function Base.sum(l::Vector{Vector{AffExpr}})
	temp = zero(l[1][1])
	for i in l
		temp += sum(i)
	end
	return temp
end

# ╔═╡ 243d047b-920d-451c-946e-1fcd45f227b4
function num2weekStr(num)
	return "W"*string(num-1)
end

# ╔═╡ 2fd96ad3-0367-49ea-846e-07f6722f82ff
@bind week Slider(1:24)

# ╔═╡ 36ca59d9-2891-43af-bdbb-0a2a4eee5853
md"""
 目前在计算的周： $(week)
"""

# ╔═╡ 9cea8136-3a62-4776-ba94-0f1aa0eaaf42
weekStr = num2weekStr(week)

# ╔═╡ 500d4a9c-0879-4738-bf22-ad726d90fe9d
@bind limit Slider(1:5)

# ╔═╡ 7ec11938-e35a-478c-9af3-5b8a8e0108b6
md"""
目前的转运商允许量：$(limit)
"""

# ╔═╡ 4669b138-a94b-43f2-a4c2-0a803f1ca3d0
@bind Transshipment_capacity Slider(6000:500:22000)

# ╔═╡ b1530cee-90e3-483d-86e8-cc54e7c24338
md"""
目前的转运能力：$(Transshipment_capacity)
"""

# ╔═╡ a9ed9185-630c-4c6e-aac5-155900de98f3
md"""
## 导入数据
"""

# ╔═╡ 4c41b978-fdd4-4b03-b621-f0753b727eb5
begin
	df_order_24_week = CSV.read("order_24_week (2).csv",DataFrame)
	df_Average_loss_of_forwarders = CSV.read("转运商平均损耗.csv",DataFrame)
	forwarders_id = df_Average_loss_of_forwarders[:,:转运商ID]
end

# ╔═╡ ef66f393-4a3b-4d0a-8d20-dd521b5529cd
filter((:材料分类,"C"),df_order_24_week,weekStr)

# ╔═╡ ccb7017d-eaaf-451e-96f0-76b729ba799c
md"""
# 线性方程求解函数
"""

# ╔═╡ 7f084915-1fb5-46aa-bc15-e1d73ed5960e
function solve_chose_forwarders(week::String,
								df_order::DataFrame,
								df_Average_loss::DataFrame,
								rate_list::Vector,
								limit::Int64,
								Transshipment_capacity::Int64)
	#每个材料类别的损耗量
	function sum_model(uⁱ,
						len_of_u::Int64,
						num_of_forwarder::Int64,
						Average_loss::Vector{Float64},
						order::Vector{Float64})
		sum_i = sum([[uⁱ[i,j] * Average_loss[j] * order[i] for i in 1:len_of_u] for j in 1:num_of_forwarder])
		return sum_i
	end

	# 全部损耗量
	function sum_model(U::Vector,
						length_list::Vector,
						num_of_forwarder::Int64,
						Average_loss::Vector{Float64},
						order_list::Vector{Vector{Float64}},
						rate_list::Vector)
		temp=0
		for (uⁱ,len_of_u,order,rate) in zip(U,length_list,order_list,rate_list)
			temp += rate* sum_model(uⁱ,
						len_of_u,
						num_of_forwarder,
						Average_loss,
						order)
		end
		return temp
	end

	#每一个供应商的转运量
	function sum_forwarders(index_of_forwarders::Int64,
							U::Vector,
							length_list::Vector,
							Average_loss::Vector{Float64},
							order_list::Vector{Vector{Float64}})
		temp = 0
		j = index_of_forwarders
		for (uⁱ,len_of_u,order) in zip(U,length_list,order_list)
			temp += sum([uⁱ[i,j] * order[i] for i in 1:len_of_u])
		end
		return temp
	end
	
	# 01约束
	function sum_bin(uⁱ,len::Int64,model)
		for i in 1:len
			@constraint(model, sum(uⁱ[i,:])<=limit)
		end
	end

	#运货量等于订货量
	function sum_bin(uⁱ,len::Int64,
					model,
					order::Vector{Float64},
					num_of_forwarder::Int64)
		for i in 1:len
			@constraint(model, sum([uⁱ[i,j] * order[i] for j in 1:num_of_forwarder]) == order[i])
		end
	end
	
	#定义模型
	model = Model(Gurobi.Optimizer)

	#定义变量
	num_of_forwarder=length(df_Average_loss_of_forwarders[:,1])
	begin
		length_A = length(filter((:材料分类,"A"),df_order,week))
		length_B = length(filter((:材料分类,"B"),df_order,week))
		length_C = length(filter((:材料分类,"C"),df_order,week))
		@variable(model, uᴬ[i = 1:length_A, j= 1:num_of_forwarder],Bin)
		@variable(model, uᴮ[i = 1:length_B, j= 1:num_of_forwarder],Bin)
		@variable(model, uᶜ[i = 1:length_C, j= 1:num_of_forwarder],Bin)
	end

	#定义目标函数
	@objective(model, Min,
			sum_model([uᴬ,uᴮ,uᶜ],
				[length_A,length_B,length_C],
				num_of_forwarder,
				df_Average_loss[:,:平均损耗],
				[filter((:材料分类,"A"),df_order,week),
					filter((:材料分类,"B"),df_order,week),
					filter((:材料分类,"C"),df_order,week)],
				rate_list))

	#定义约束
	#01约束
	for (uⁱ,len_of_u) in zip([uᴬ,uᴮ,uᶜ],[length_A,length_B,length_C])
		sum_bin(uⁱ,len_of_u,model)
	end
	#供应商的转运量小于6000
	for j in 1:num_of_forwarder
		@constraint(model,
		sum_forwarders(j,[uᴬ,uᴮ,uᶜ],[15,14,15],
					df_Average_loss[:,:平均损耗],
					[filter((:材料分类,"A"),df_order,week),
					filter((:材料分类,"B"),df_order,week),
					filter((:材料分类,"C"),df_order,week)])
		<=Transshipment_capacity)
	end
	#运货量等于订货量
	for (uⁱ,len_of_u,order) in zip([uᴬ,uᴮ,uᶜ],
							[length_A,length_B,length_C],
							[filter((:材料分类,"A"),df_order,week),
								filter((:材料分类,"B"),df_order,week),
								filter((:材料分类,"C"),df_order,week)])
		sum_bin(uⁱ,len_of_u,model,order,num_of_forwarder)
	end
	optimize!(model)
	return model,(uᴬ,uᴮ,uᶜ)
end

# ╔═╡ 407fe9e3-c524-4b51-a0dd-0229e398f535
function solve_chose_forwarders(week::String,
								df_order::DataFrame,
								df_Average_loss::DataFrame)
	solve_chose_forwarders(weekStr,df_order,df_Average_loss,[1.2,1.1,1,1,1],limit,Transshipment_capacity)
end

# ╔═╡ 3737fa99-4b4d-46f3-bdd4-106e589aa847
(model,var) = solve_chose_forwarders(weekStr,df_order_24_week,df_Average_loss_of_forwarders)

# ╔═╡ febc35df-f986-43e6-a98a-6a8107ef7dd9
termination_status(model)

# ╔═╡ 1f51645c-fa66-45d0-a2a3-846554264525
termination_status(model)

# ╔═╡ 14a0e20c-b79d-4d0f-aed9-e31753f2ddfa
md"""
# 把解转化为坐标
"""

# ╔═╡ 02bd0f1c-4c41-4a3a-8928-d578cccec15a
# 解矩阵转化为数字
function mat2sym(mat::Matrix)
	
	function var2sym(sym_list::Vector,
					 var::Vector)
		temp = missing
		for (i,sym) in zip(var,sym_list)
			if i == 1.0
				temp = sym
				break
			end
		end
		return temp
	end
	
	function mat2vec(mat::Matrix)
		x,_ = size(mat)
		return [mat[i,:] for i in 1:x]
	end
	
	return mat |> x->value.(x) |> mat2vec |> x->(x-> var2sym(forwarders_id,x)).(x)
end

# ╔═╡ 9851ee35-f131-48e7-b87e-1c03d27b5263
# 解转化为坐标数字元组
function solve2location(供应商ID表_list::Vector,
						solve_list::Tuple,
						data_list::Vector)
	#辅助函数
	function delete(str::AbstractString,
					del::String)
		return replace(str,del=>"")
	end

	function sym2num(str::AbstractString)
		return str |> x->delete(x,"T") |> x->delete(x,"S") |> x->Parsers.parse(Int64,x)
	end

	function sym2num(str::Missing)
		return -1
	end	

	function mat2location(供应商ID表::Vector,
							solve::Matrix,
							data::Vector)
		x = 供应商ID表 |> x->sym2num.(x)
		y = solve |> mat2sym |> x->sym2num.(x)
		return zip(x,y,data)
	end
	
	temp = []
	for (供应商ID表,solve,data) in zip(供应商ID表_list,solve_list,data_list)
		output = mat2location(供应商ID表,solve,data)
		for i in output
			temp = cat(temp,i,dims=1)
		end
	end
	return temp
end

# ╔═╡ a083a5da-0536-4bd9-9c5f-2ec66678d7f2
begin
	供应商ID表_list = [filter((:材料分类,i),df_order_24_week)[:,:Column1] for i in ["A","B","C"]]
	周订购数据_list = [filter((:材料分类,i),df_order_24_week)[:,week+2] for i in ["A","B","C"]]
	location_list = solve2location(供应商ID表_list, var, 周订购数据_list)
end

# ╔═╡ 393930a9-26f6-4a25-9648-97b90d21eb51
md"""
# 写入表格，准备复制
"""

# ╔═╡ c9dee99e-9a55-4092-b2e4-e48dc1dcccba
#temp = DataFrame(fill!(Matrix{Float64}(undef, 402, 8),-114.514), :auto)

# ╔═╡ c7764ec3-6fc2-455a-8fdf-b1a319d4c0f6
for i in location_list
	(y,x,data) = i
	if x == -1
		continue
	end
	temp[y,x] =data
end

# ╔═╡ a9df1cd9-bfe1-461d-bac3-a91589c2163f
CSV.write("temp_location-$(weekStr).csv",temp)

# ╔═╡ 7e218375-87e3-44a7-afa3-d5f4052375fc
#read("temp_location-$(weekStr).csv",String) |> x->replace(x,"-114.514"=>"") |> x-> write("temp_location-$(weekStr).csv",x)

# ╔═╡ 606088ae-d999-42ea-aa98-308412ac779d
function write_table(location_list::Vector,weekStr::String)
	temp = DataFrame(fill!(Matrix{Float64}(undef, 402, 8),-114.514), :auto)
	for i in location_list
		(y,x,data) = i
		if x == -1
			continue
		end
		temp[y,x] =data
	end
	path = joinpath("location","temp_location-$(weekStr).csv")
	CSV.write(path,temp)
	read(path,String) |> x->replace(x,"-114.514"=>"") |> x-> write(path,x)
end

# ╔═╡ 2927eb71-9b9e-4026-8857-3331e715fc39
joinpath("location","temp_location-$(weekStr).csv")

# ╔═╡ 618e342c-f035-4337-8705-91c10eaffbfd
write_table(location_list,weekStr)

