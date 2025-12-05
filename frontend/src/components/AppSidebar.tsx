import { Home, UtensilsCrossed, Heart, Settings, LogOut, User } from "lucide-react"
import {
  Sidebar,
  SidebarContent, 
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  useSidebar,
} from "@/components/ui/sidebar"
import { useAuth } from "@/hooks/useAuth"
import { signOut } from "@/lib/supabase/auth"
import { useNavigate, useLocation, Link } from "react-router"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

const menuItems = [
  {
    title: "Dashboard",
    url: "/dashboard",
    icon: Home,
  },
  {
    title: "Recommendations",
    url: "/recommendations",
    icon: UtensilsCrossed,
  },
  {
    title: "Favorites",
    url: "/favorites",
    icon: Heart,
  },
  {
    title: "Settings",
    url: "/settings",
    icon: Settings,
  }
]

export function AppSidebar() {
  const { profile, user } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const { state } = useSidebar()
  const handleSignOut = async () => {
    await signOut()
    navigate("/auth/login")
  }

  return (
    <Sidebar variant="inset" collapsible="icon" >
      <SidebarHeader>
        <div className="flex items-center gap-2 px-2 py-4">
          <img src="/MYF_logo.png" alt="MYF.ai" className="h-6 w-6" />
          {state === "expanded" && (
            <span className="font-bold text-lg">MYF.ai</span>
          )}
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {menuItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    isActive={location.pathname === item.url}
                  >
                    <Link to={item.url}>
                      <item.icon />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton>
                  <User />
                  <span>{profile?.email}</span>
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent side="top" className="w-[--radix-popper-anchor-width]">
                <DropdownMenuItem disabled>
                  <span className="text-xs text-muted-foreground">{user?.email}</span>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={handleSignOut}>
                  <LogOut className="mr-2 h-4 w-4" />
                  <span>Sign out</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  )
}
