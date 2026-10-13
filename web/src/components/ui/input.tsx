import * as React from "react"

import { cn } from "@/lib/utils"

const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<"input">>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-8 w-full rounded-sm border border-[#DCD9D0] bg-[#FBFBF9] px-2.5 py-1 text-xs text-[#231F20] transition-colors file:border-0 file:bg-transparent file:text-xs file:font-medium placeholder:text-[#6B6661] focus-visible:outline-none focus-visible:border-[#231F20] disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Input.displayName = "Input"

export { Input }
